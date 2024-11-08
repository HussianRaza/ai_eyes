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
        """
        Initialize with input path(s) and output directory

        Args:
            input_path: Single path or list of paths to process
            output_dir: Directory to save outputs
        """
        self.input_paths = [input_path] if isinstance(input_path, str) else input_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_file_type(self, file_path: str) -> str:
        """Determine if file is image or video"""
        ext = Path(file_path).suffix.lower()
        if ext in self.IMAGE_EXTENSIONS:
            return 'image'
        elif ext in self.VIDEO_EXTENSIONS:
            return 'video'
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def generate_output_path(self, input_path: str) -> str:
        """Generate output path with same extension as input"""
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
        """Update metrics with new frame processing time"""
        self.processed_frames += 1
        self.processing_times.append(frame_time)

    def get_stats(self) -> Dict:
        """Get current processing statistics"""
        if not self.processing_times:
            return {}

        elapsed_time = time.time() - self.start_time
        return {
            'total_frames': self.processed_frames,
            'average_fps': self.processed_frames / elapsed_time,
            'average_frame_time': np.mean(self.processing_times),
            'total_time': elapsed_time
        }

class FrameProcessor:
    def __init__(self, currency_model: YOLO, object_model: YOLO, batch_size: int = 4):
        self.currency_model = currency_model
        self.object_model = object_model
        self.batch_size = batch_size
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue()
        self.processing = True
        self.metrics = ProcessingMetrics()
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
                    # Process currency detection
                    currency_results = self.currency_model(frames_batch, verbose=False)

                    # Create masked frames batch
                    masked_frames = []
                    for frame, curr_result in zip(frames_batch, currency_results):
                        mask = create_currency_mask(frame, [curr_result])
                        masked_frame = frame.copy()
                        masked_frame[mask > 0] = 0
                        masked_frames.append(masked_frame)

                    # Process object detection
                    object_results = self.object_model(masked_frames, verbose=False)

                # Process results for each frame
                for frame, curr_result, obj_result in zip(frames_batch, currency_results, object_results):
                    frame_time = (time.time() - batch_start) / len(frames_batch)
                    self.metrics.update(frame_time)

                    processed_frame, results = self._process_single_frame_results(
                        frame, curr_result, obj_result)
                    self.result_queue.put((processed_frame, results))

    def _process_single_frame_results(self, frame, currency_result, object_result):
        result_frame = frame.copy()
        results = {'currency': [], 'objects': []}

        # Process currency detections
        if currency_result.boxes is not None:
            result_frame = draw_detection_labels(result_frame, [currency_result], (0, 255, 0))
            for box in currency_result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                results['currency'].append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(box.conf[0]),
                    'class': currency_result.names[int(box.cls[0])]
                })

        # Process object detections
        if object_result.boxes is not None:
            result_frame = draw_detection_labels(result_frame, [object_result], (255, 0, 0))
            for box in object_result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                results['objects'].append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(box.conf[0]),
                    'class': object_result.names[int(box.cls[0])]
                })

        return result_frame, results

    def process_image(self, image_path: str, output_path: str, display_result: bool = True) -> Dict:
        """Process a single image and save result"""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        self.add_frame(frame)
        result = self.get_result()

        if result:
            processed_frame, detection_results = result

            # Save the processed image
            cv2.imwrite(output_path, processed_frame)

            # Display if requested
            if display_result:
                cv2_imshow(processed_frame)

            return {
                'path': output_path,
                'detections': detection_results,
                'metrics': self.metrics.get_stats()
            }
        return None

    def process_video(self, video_path: str, output_path: str, display_frames: bool = True) -> Dict:
        """Process a video file and save result"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.metrics.total_frames = total_frames

        # Initialize video writer
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        all_detections = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                self.add_frame(frame)
                result = self.get_result()

                if result:
                    processed_frame, detection_results = result

                    # Save frame to video
                    out.write(processed_frame)

                    # Store detections
                    all_detections.append(detection_results)

                    # Display if requested
                    if display_frames:
                        cv2_imshow(processed_frame)

                    # Print progress
                    progress = (self.metrics.processed_frames / total_frames) * 100
                    print(f"\rProgress: {progress:.1f}%", end="")

        finally:
            cap.release()
            out.release()
            print("\nVideo processing complete")

        return {
            'path': output_path,
            'detections': all_detections,
            'metrics': self.metrics.get_stats()
        }

    def add_frame(self, frame):
        try:
            self.frame_queue.put(frame, timeout=1.0)
        except queue.Full:
            print("Warning: Frame queue full, skipping frame")

    def get_result(self):
        try:
            return self.result_queue.get(timeout=1.0)
        except queue.Empty:
            return None

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
    Main function with improved file handling
    """
    print("\n=== Starting Application ===")

    try:
        # Load models
        currency_model, object_model = load_models(
            currency_model_path,
            object_model_path
        )

        # Define input paths (can be single file or directory)
        input_paths = [
            # Add more paths as needed
            "/content/10, 20, 50 & 100 Rupay ko Pehchano - State Bank of Pakistan (1080p, h264, youtube).mp4"

        ]

        # Define output directory
        output_dir = "/content/output"

        # Process all files
        results = process_files(
            input_paths=input_paths,
            output_dir=output_dir,
            currency_model=currency_model,
            object_model=object_model,
            batch_size=12,
            display_results=True
        )

        # Print processing statistics
        print("\n=== Processing Statistics ===")
        for input_path, result in results.items():
            print(f"\nFile: {input_path}")
            print(f"Output: {result['path']}")
            print("Metrics:", result['metrics'])

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