

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