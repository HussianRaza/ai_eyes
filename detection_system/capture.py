# detection_system/capture.py
import cv2
import numpy as np
import time
from typing import Optional
from IPython.display import display, Javascript, clear_output
from google.colab.patches import cv2_imshow
from base64 import b64decode
from google.colab.output import eval_js

class WebcamCapture:
    def __init__(self):
        js_code = """
            // Define elements in global scope
            window.video = document.createElement('video');
            window.canvas = document.createElement('canvas');
            
            // Define functions in global scope
            window.startWebcam = async function() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                    window.video.srcObject = stream;
                    window.video.style.display = 'none';
                    window.canvas.style.display = 'none';
                    document.body.appendChild(window.video);
                    document.body.appendChild(window.canvas);
                    await window.video.play();
                    window.canvas.width = window.video.videoWidth;
                    window.canvas.height = window.video.videoHeight;
                    return true;
                } catch(err) {
                    console.error("Error starting webcam:", err);
                    return false;
                }
            }
            
            window.captureFrame = function() {
                try {
                    window.canvas.getContext('2d').drawImage(window.video, 0, 0);
                    return window.canvas.toDataURL('image/jpeg');
                } catch(err) {
                    console.error("Error capturing frame:", err);
                    return null;
                }
            }
            
            window.stopWebcam = function() {
                try {
                    const stream = window.video.srcObject;
                    const tracks = stream.getTracks();
                    tracks.forEach(track => track.stop());
                    window.video.remove();
                    window.canvas.remove();
                    return true;
                } catch(err) {
                    console.error("Error stopping webcam:", err);
                    return false;
                }
            }
        """
        self.js = Javascript(js_code)
        
    def start(self) -> None:
        display(self.js)
        eval_js('startWebcam()')
        time.sleep(3)  # Give time for webcam to initialize
        
    def read_frame(self) -> Optional[np.ndarray]:
        try:
            frame_data = eval_js('captureFrame()')
            if frame_data is None:
                return None
                
            # Convert JS data to OpenCV image
            image_bytes = b64decode(frame_data.split(',')[1])
            jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(jpg_as_np, flags=1)
            return frame
        except Exception as e:
            print(f"Error reading frame: {str(e)}")
            return None
        
    def stop(self) -> None:
        try:
            eval_js('stopWebcam()')
        except Exception as e:
            print(f"Error stopping webcam: {str(e)}")