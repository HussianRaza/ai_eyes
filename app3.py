import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple, List, Dict, Union, Optional
import torch
import time
from google.colab.patches import cv2_imshow
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import os
from pathlib import Path
import glob

class FileProcessor:
    """Handles file operations and determines processing type"""

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

    def __init__(self, input_path: Union[str, List[str]], output_dir: str):
        self.input_paths = [input_path] if isinstance(input_path, str) else input_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_file_type(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        if ext in self.IMAGE_EXTENSIONS:
            return 'image'
        elif ext in self.VIDEO_EXTENSIONS:
            return 'video'
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def generate_output_path(self, input_path: str) -> str:
        input_path = Path(input_path)
        return str(self.output_dir / f"processed_{input_path.stem}{input_path.suffix}")

class ProcessingMetrics:
    """Tracks processing metrics and performance"""

    def __init__(self):
        self.start_time = time.time()
        self.processed_frames = 0
        self.total_frames = 0
        self.processing_times = []

    def update(self, frame_time: float):
        self.processed_frames += 1
        self.processing_times.append(frame_time)

    def get_stats(self) -> Dict:
        if not self.processing_times:
            return {}

        elapsed_time = time.time() - self.start_time
        return {
            'total_frames': self.processed_frames,
            'average_fps': self.processed_frames / elapsed_time,
            'average_frame_time': np.mean(self.processing_times),
            'total_time': elapsed_time
        }

class AudioAlert:
    """Handles audio alerts for proximity warnings"""
    
    def __init__(self):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.last_alert_time = 0
            self.MIN_ALERT_INTERVAL = 3.0
        except ImportError:
            print("pyttsx3 not installed. Audio alerts will be disabled.")
            self.engine = None
    
    def speak(self, message: str):
        current_time = time.time()
        if self.engine and (current_time - self.last_alert_time) >= self.MIN_ALERT_INTERVAL:
            self.engine.say(message)
            self.engine.runAndWait()
            self.last_alert_time = current_time

def create_currency_mask(frame: np.ndarray, currency_results: List) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    for result in currency_results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    return mask

def draw_detection_labels(
    frame: np.ndarray,
    results: List,
    color: Tuple[int, int, int]
) -> np.ndarray:
    annotated_frame = frame.copy()
    
    for result in results:
        if result.boxes is not None:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                class_name = result.names[int(cls)]
                confidence = float(conf)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name} {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1),
                            color, -1)
                cv2.putText(annotated_frame, label, 
                          (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated_frame

def estimate_distance(bbox: List[int], frame_height: int) -> float:
    KNOWN_HEIGHT = 0.5  # Known height of reference object in meters
    KNOWN_DISTANCE = 1.0  # Known distance to reference object in meters
    KNOWN_PIXELS = frame_height * 0.4  # Known pixels at reference distance
    
    focal_length = (KNOWN_PIXELS * KNOWN_DISTANCE) / KNOWN_HEIGHT
    
    _, y1, _, y2 = bbox
    object_height_pixels = y2 - y1
    
    distance = (KNOWN_HEIGHT * focal_length) / object_height_pixels
    
    return round(distance, 2)

def get_proximity_alert(distance: float) -> Tuple[bool, str]:
    if distance < 1.0:
        return True, "Warning: Object very close, less than 1 meter away"
    elif distance < 2.0:
        return True, "Caution: Object approaching, approximately 2 meters away"
    return False, ""

def load_models(currency_model_path: str, object_model_path: str) -> Tuple[YOLO, YOLO]:
    try:
        print("Loading currency detection model...")
        currency_model = YOLO(currency_model_path)
        
        print("Loading object detection model...")
        object_model = YOLO(object_model_path)
        
        if torch.cuda.is_available():
            currency_model = currency_model.to('cuda')
            object_model = object_model.to('cuda')
            print("Models loaded on GPU")
        else:
            print("Models loaded on CPU")
            
        return currency_model, object_model
        
    except Exception as e:
        raise RuntimeError(f"Error loading models: {str(e)}")

class FrameProcessor:
    """Handles frame processing with batch support and multi-threading"""

    def __init__(self, currency_model: YOLO, object_model: YOLO, batch_size: int = 4):
        self.currency_model = currency_model
        self.object_model = object_model
        self.batch_size = batch_size
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue()
        self.processing = True
        self.metrics = ProcessingMetrics()
        self.audio_alert = AudioAlert()
        self.batch_processing_thread = threading.Thread(target=self._process_batches)
        self.batch_processing_thread.start()

    def _process_batches(self):
        while self.processing:
            frames_batch = []
            while len(frames_batch) < self.batch_size and self.processing:
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                    frames_batch.append(frame)
                except queue.Empty:
                    break

            if frames_batch:
                batch_start = time.time()

                with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                    currency_results = self.currency_model(frames_batch, verbose=False)

                    masked_frames = []
                    for frame, curr_result in zip(frames_batch, currency_results):
                        mask = create_currency_mask(frame, [curr_result])
                        masked_frame = frame.copy()
                        masked_frame[mask > 0] = 0
                        masked_frames.append(masked_frame)

                    object_results = self.object_model(masked_frames, verbose=False)

                for frame, curr_result, obj_result in zip(frames_batch, currency_results, object_results):
                    frame_time = (time.time() - batch_start) / len(frames_batch)
                    self.metrics.update(frame_time)

                    processed_frame, results = self._process_single_frame_results(
                        frame, curr_result, obj_result)
                    self.result_queue.put((processed_frame, results))

    def _process_single_frame_results(self, frame, currency_result, object_result):
        result_frame = frame.copy()
        frame_height = frame.shape[0]
        results = {'currency': [], 'objects': []}

        if currency_result.boxes is not None:
            result_frame = draw_detection_labels(result_frame, [currency_result], (0, 255, 0))
            for box in currency_result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                results['currency'].append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(box.conf[0]),
                    'class': currency_result.names[int(box.cls[0])]
                })

        if object_result.boxes is not None:
            result_frame = draw_detection_labels(result_frame, [object_result], (255, 0, 0))
            for box in object_result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                distance = estimate_distance([x1, y1, x2, y2], frame_height)
                results['objects'].append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(box.conf[0]),
                    'class': object_result.names[int(box.cls[0])],
                    'distance': distance
                })

                alert_needed, message = get_proximity_alert(distance)
                if alert_needed:
                    self.audio_alert.speak(message)

        return result_frame, results

    def process_image(self, input_path: str, output_path: str, display_results: bool = True) -> Dict:
        frame = cv2.imread(input_path)
        self.frame_queue.put(frame)
        
        processed_frame, results = self.result_queue.get()
        
        cv2.imwrite(output_path, processed_frame)
        
        if display_results:
            cv2_imshow(processed_frame)

        return results

    def process_video(self, input_path: str, output_path: str, display_results: bool = False) -> Dict:
        cap = cv2.VideoCapture(input_path)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        results = {'frames': []}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_queue.put(frame)

            processed_frame, frame_results = self.result_queue.get()
            results['frames'].append(frame_results)

            out.write(processed_frame)
            
            if display_results:
                cv2_imshow(processed_frame)

        cap.release()
        out.release()
        
        return results

    def stop(self):
        self.processing = False
        self.batch_processing_thread.join()

def process_files(
    input_paths: Union[str, List[str]],
    output_dir: str,
    currency_model: YOLO,
    object_model: YOLO,
    batch_size: int = 4,
    display_results: bool = True
) -> Dict[str, Dict]:
    """
    Process multiple files and save results

    Args:
        input_paths: Single path or list of paths to process
        output_dir: Directory to save outputs
        currency_model: YOLO model for currency detection
        object_model: YOLO model for object detection
        batch_size: Batch size for processing
        display_results: Whether to display results

    Returns:
        Dictionary containing processing results for each file
    """
    file_processor = FileProcessor(input_paths, output_dir)
    frame_processor = FrameProcessor(currency_model, object_model, batch_size)
    results = {}

    try:
        for input_path in file_processor.input_paths:
            print(f"\nProcessing: {input_path}")

            # Generate output path
            output_path = file_processor.generate_output_path(input_path)

            # Process based on file type
            file_type = file_processor.get_file_type(input_path)
            if file_type == 'image':
                results[input_path] = frame_processor.process_image(
                    input_path, output_path, display_results)
            else:  # video
                results[input_path] = frame_processor.process_video(
                    input_path, output_path, display_results)

            print(f"Saved output to: {output_path}")

    finally:
        frame_processor.stop()

    return results

    
    



def main():
    """
    Main function with improved file handling and error management
    """
    print("\n=== Starting Application ===")

    try:
        # Configure model paths - update these paths according to your setup
        currency_model_path = "path/to/currency_model.pt"  # Update this path
        object_model_path = "path/to/object_model.pt"      # Update this path

        # Load models
        currency_model, object_model = load_models(
            currency_model_path,
            object_model_path
        )

        # Define input paths (can be single file or directory)
        input_paths = [
            "path/to/your/input/file.mp4",  # Update with your input path
            # Add more paths as needed
        ]

        # Define output directory
        output_dir = "path/to/output/directory"  # Update with your output directory

        # Process all files
        results = process_files(
            input_paths=input_paths,
            output_dir=output_dir,
            currency_model=currency_model,
            object_model=object_model,
            batch_size=12,  # Adjust based on your GPU memory
            display_results=True
        )

        # Print processing statistics
        print("\n=== Processing Statistics ===")
        for input_path, result in results.items():
            if result:  # Check if result exists
                print(f"\nFile: {input_path}")
                print(f"Output: {result['path']}")
                print("Metrics:", result['metrics'])
                
                # Print detection counts
                if 'detections' in result:
                    if isinstance(result['detections'], list):
                        # For videos (multiple frames)
                        total_currency = sum(len(frame['currency']) for frame in result['detections'])
                        total_objects = sum(len(frame['objects']) for frame in result['detections'])
                        print(f"Total currency detections: {total_currency}")
                        print(f"Total object detections: {total_objects}")
                    else:
                        # For single images
                        print(f"Currency detections: {len(result['detections']['currency'])}")
                        print(f"Object detections: {len(result['detections']['objects'])}")

    except Exception as e:
        print(f"\n❌ Error in main application: {str(e)}")
        raise
    finally:
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== Application Finished ===")

if __name__ == "__main__":
    main()