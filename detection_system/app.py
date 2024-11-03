# detection_system/app.py
import time
import cv2
from google.colab.patches import cv2_imshow
from IPython.display import clear_output 
from detection_system.models import DetectionModel
from detection_system.processor import FrameProcessor
from detection_system.capture import WebcamCapture

class DetectionApp:
    def __init__(
        self,
        currency_model_path: str,
        object_model_path: str,
        conf_threshold: float = 0.25
    ):
        self.currency_model = DetectionModel(currency_model_path)
        self.object_model = DetectionModel(object_model_path)
        self.processor = FrameProcessor(
            self.currency_model,
            self.object_model,
            conf_threshold
        )
        
    def run_detection(self, fps_target: int = 10) -> None:
        """Run real-time currency and object detection on webcam feed"""
        capture = WebcamCapture()
        print("Initializing webcam...")
        capture.start()
        
        try:
            frame_count = 0
            last_time = time.time()
            fps = 0
            
            while True:
                try:
                    frame = capture.read_frame()
                    if frame is None:
                        print("Error reading frame. Retrying...")
                        time.sleep(0.1)
                        continue
                        
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
                    
                    print(f"\nFPS: {fps:.1f}")
                    
                    frame_count += 1
                    
                    # Control frame rate
                    time.sleep(max(0, 1/fps_target - (time.time() - current_time)))
                    
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    time.sleep(0.1)
                    continue
                    
        except KeyboardInterrupt:
            print("\nStopping detection...")
        finally:
            capture.stop()