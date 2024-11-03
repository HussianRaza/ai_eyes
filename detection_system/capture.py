

# detection_system/capture.py
import cv2
import numpy as np
import time
from typing import Optional, Tuple
from IPython.display import display, Javascript, clear_output
from google.colab.patches import cv2_imshow
from base64 import b64decode
from google.colab.output import eval_js

class WebcamCapture:
    def __init__(self):
        self._js = self._create_js()


    def _create_js(self) -> Javascript:
        return Javascript('''
        async function initWebcam() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                const video = document.createElement('video');
                const canvas = document.createElement('canvas');
                
                video.srcObject = stream;
                document.body.appendChild(video);
                document.body.appendChild(canvas);
                
                // Hide the video element
                video.style.display = 'none';
                canvas.style.display = 'none';
                
                // Wait for video to start playing
                await video.play();
                
                // Set canvas size to match video
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                
                window.streamReference = stream;
                window.videoReference = video;
                window.canvasReference = canvas;
            } catch(err) {
                console.error("Error initializing webcam:", err);
            }
        }
        
        function captureWebcam() {
            try {
                const video = window.videoReference;
                const canvas = window.canvasReference;
                const context = canvas.getContext('2d');
                
                context.drawImage(video, 0, 0);
                return canvas.toDataURL('image/jpeg');
            } catch(err) {
                console.error("Error capturing frame:", err);
                return null;
            }
        }
        
        function stopWebcam() {
            try {
                const stream = window.streamReference;
                const tracks = stream.getTracks();
                tracks.forEach(track => track.stop());
            } catch(err) {
                console.error("Error stopping webcam:", err);
            }
        }
    ''')
        
    
        
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