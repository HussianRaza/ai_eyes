# detection_system/models.py
from dataclasses import dataclass
from typing import List, Tuple, Dict, Union, Optional
import numpy as np
from ultralytics import YOLO

@dataclass
class Detection:
    bbox: List[int]
    confidence: float
    class_name: str

@dataclass
class FrameResults:
    currency: List[Detection]
    objects: List[Detection]

class DetectionModel:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.names = self.model.names
        
    def predict(self, frame: np.ndarray, conf_threshold: float) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            conf=conf_threshold,
            verbose=False
        )
        
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.names[class_id]
                
                detections.append(Detection(
                    bbox=[x1, y1, x2, y2],
                    confidence=conf,
                    class_name=class_name
                ))
        
        return detections

# detection_system/visualization.py
import cv2
import numpy as np
from .models import Detection, FrameResults

class Visualizer:
    def __init__(self):
        self.currency_color = (0, 255, 0)  # Green
        self.object_color = (255, 0, 0)    # Blue
        
    def draw_detection(
        self,
        frame: np.ndarray,
        detection: Detection,
        color: Tuple[int, int, int]
    ) -> None:
        x1, y1, x2, y2 = detection.bbox
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        label = f"{detection.class_name}: {detection.confidence:.2f}"
        label_y = y1 - 10 if y1 > 20 else y1 + 10
        
        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame,
                     (x1, label_y - h - 5),
                     (x1 + w, label_y + 5),
                     color,
                     -1)
        
        # Draw label text
        cv2.putText(frame,
                    label,
                    (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2)
    
    def draw_fps(self, frame: np.ndarray, fps: float) -> None:
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
    
    def visualize_frame(
        self,
        frame: np.ndarray,
        results: FrameResults,
        fps: Optional[float] = None
    ) -> np.ndarray:
        result_frame = frame.copy()
        
        # Draw currency detections
        for detection in results.currency:
            self.draw_detection(result_frame, detection, self.currency_color)
            
        # Draw object detections
        for detection in results.objects:
            self.draw_detection(result_frame, detection, self.object_color)
            
        # Draw FPS if provided
        if fps is not None:
            self.draw_fps(result_frame, fps)
            
        return result_frame

# detection_system/processor.py
import cv2
import numpy as np
from typing import Tuple
from .models import DetectionModel, FrameResults
from .visualization import Visualizer

class FrameProcessor:
    def __init__(
        self,
        currency_model: DetectionModel,
        object_model: DetectionModel,
        conf_threshold: float = 0.25
    ):
        self.currency_model = currency_model
        self.object_model = object_model
        self.conf_threshold = conf_threshold
        self.visualizer = Visualizer()
        
    def create_currency_mask(
        self,
        frame: np.ndarray,
        currency_detections: List[Detection]
    ) -> np.ndarray:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for detection in currency_detections:
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            
        return mask
    
    def process_frame(
        self,
        frame: np.ndarray,
        debug: bool = True
    ) -> Tuple[np.ndarray, FrameResults]:
        # Detect currency
        currency_detections = self.currency_model.predict(
            frame,
            self.conf_threshold
        )
        
        # Create mask for currency regions
        currency_mask = self.create_currency_mask(frame, currency_detections)
        
        # Mask frame for object detection
        masked_frame = frame.copy()
        masked_frame[currency_mask > 0] = 0
        
        # Detect objects
        object_detections = self.object_model.predict(
            masked_frame,
            self.conf_threshold
        )
        
        # Compile results
        results = FrameResults(
            currency=currency_detections,
            objects=object_detections
        )
        
        # Visualize results
        if debug:
            debug_mask = cv2.cvtColor(currency_mask, cv2.COLOR_GRAY2BGR)
            debug_mask[currency_mask > 0] = [0, 255, 0]
            processed_frame = cv2.addWeighted(
                self.visualizer.visualize_frame(frame, results),
                0.7,
                debug_mask,
                0.3,
                0
            )
        else:
            processed_frame = self.visualizer.visualize_frame(frame, results)
            
        return processed_frame, results

# detection_system/capture.py
import cv2
import numpy as np
import time
from typing import Optional, Tuple
from IPython.display import display, Javascript, clear_output
from google.colab.patches import cv2_imshow
from base64 import b64decode

class WebcamCapture:
    def __init__(self):
        self._js = self._create_js()
        
    def _create_js(self) -> Javascript:
        # Your existing Javascript code here
        # (The long Javascript block from the original code)
        pass
        
    def start(self) -> None:
        display(self._js)
        display(Javascript('initWebcam()'))
        time.sleep(3)
        
    def read_frame(self) -> Optional[np.ndarray]:
        frame_data = eval_js('captureWebcam()')
        if frame_data is None:
            return None
            
        # Convert JS data to OpenCV image
        image_bytes = b64decode(frame_data.split(',')[1])
        jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, flags=1)
        return frame
        
    def stop(self) -> None:
        display(Javascript('stopWebcam()'))

class VideoCapture:
    def __init__(self, source: str):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
            
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()
        
    def stop(self) -> None:
        self.cap.release()

# detection_system/app.py
import time
from typing import Optional
from .models import DetectionModel
from .processor import FrameProcessor
from .capture import WebcamCapture, VideoCapture

class DetectionApp:
    def __init__(
        self,
        currency_model_path: str,
        object_model_path: str,
        conf_threshold: float = 0.25
    ):
        # Initialize models
        self.currency_model = DetectionModel(currency_model_path)
        self.object_model = DetectionModel(object_model_path)
        
        # Initialize processor
        self.processor = FrameProcessor(
            self.currency_model,
            self.object_model,
            conf_threshold
        )
        
    def process_webcam(
        self,
        fps_target: int = 10
    ) -> None:
        capture = WebcamCapture()
        capture.start()
        
        try:
            frame_count = 0
            last_time = time.time()
            fps = 0
            
            while True:
                frame = capture.read_frame()
                if frame is None:
                    break
                    
                # Process frame
                processed_frame, results = self.processor.process_frame(frame)
                
                # Calculate FPS
                current_time = time.time()
                if current_time - last_time > 1.0:
                    fps = frame_count / (current_time - last_time)
                    frame_count = 0
                    last_time = current_time
                
                # Display frame
                clear_output(wait=True)
                cv2_imshow(processed_frame)
                
                # Print detections
                self._print_results(results)
                
                frame_count += 1
                
                # Control frame rate
                time.sleep(max(0, 1/fps_target - (time.time() - current_time)))
                
        except KeyboardInterrupt:
            print("\nStopping webcam processing...")
        finally:
            capture.stop()
            
    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> None:
        capture = VideoCapture(input_path)
        
        # Initialize video writer if needed
        writer = None
        if output_path:
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                capture.fps,
                (capture.frame_width, capture.frame_height)
            )
        
        try:
            frame_count = 0
            while True:
                ret, frame = capture.read_frame()
                if not ret:
                    break
                    
                frame_count += 1
                processed_frame, results = self.processor.process_frame(frame)
                
                # Display frame
                cv2_imshow(processed_frame)
                
                # Save frame if requested
                if writer:
                    writer.write(processed_frame)
                    
                # Print detections
                self._print_results(results)
                
                time.sleep(0.1)  # Small delay between frames
                
        finally:
            capture.stop()
            if writer:
                writer.release()
                
    def _print_results(self, results: FrameResults) -> None:
        if results.currency or results.objects:
            print("\nDetections:")
            if results.currency:
                print("Currency:")
                for det in results.currency:
                    print(f"  - {det.class_name} (Confidence: {det.confidence:.2f})")
                    
            if results.objects:
                print("Objects:")
                for det in results.objects:
                    print(f"  - {det.class_name} (Confidence: {det.confidence:.2f})")