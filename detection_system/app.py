

# detection_system/app.py
import time
import cv2
from typing import Optional
from google.colab.patches import cv2_imshow
from IPython.display import clear_output 
from detection_system.models import DetectionModel , FrameResults
from detection_system.processor import FrameProcessor
from detection_system.capture import WebcamCapture, VideoCapture

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