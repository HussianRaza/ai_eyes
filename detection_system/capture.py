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
            var video = document.createElement('video');
            var canvas = document.createElement('canvas');
            
            async function startWebcam() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                    video.srcObject = stream;
                    video.style.display = 'none';
                    canvas.style.display = 'none';
                    document.body.appendChild(video);
                    document.body.appendChild(canvas);
                    await video.play();
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    return true;
                } catch(err) {
                    console.error("Error starting webcam:", err);
                    return false;
                }
            }
            
            async function captureFrame() {
                try {
                    canvas.getContext('2d').drawImage(video, 0, 0);
                    return canvas.toDataURL('image/jpeg');
                } catch(err) {
                    console.error("Error capturing frame:", err);
                    return null;
                }
            }
            
            async function stopWebcam() {
                try {
                    const stream = video.srcObject;
                    const tracks = stream.getTracks();
                    tracks.forEach(track => track.stop());
                    video.remove();
                    canvas.remove();
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
        frame_data = eval_js('captureFrame()')
        if frame_data is None:
            return None
            
        # Convert JS data to OpenCV image
        image_bytes = b64decode(frame_data.split(',')[1])
        jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, flags=1)
        return frame
        
    def stop(self) -> None:
        eval_js('stopWebcam()')