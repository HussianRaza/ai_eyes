# detection_system/capture.py

import cv2
import numpy as np
import time
import logging
from typing import Optional
from IPython.display import display, Javascript, clear_output
from google.colab.patches import cv2_imshow
from base64 import b64decode
from google.colab.output import eval_js

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebcamCapture:
    def __init__(self):
        self.is_initialized = False
        
    def start(self) -> bool:
        """Initialize and start the webcam capture."""
        js_code = """
        async function initializeWebcam() {
            // Clean up any existing elements
            const cleanup = () => {
                const existingVideo = document.querySelector('#captureVideo');
                const existingCanvas = document.querySelector('#captureCanvas');
                if (existingVideo) existingVideo.remove();
                if (existingCanvas) existingCanvas.remove();
            };
            
            cleanup();
            
            // Create new elements
            const video = document.createElement('video');
            const canvas = document.createElement('canvas');
            
            // Set IDs for reliable querying
            video.id = 'captureVideo';
            canvas.id = 'captureCanvas';
            
            // Hide elements
            video.style.display = 'none';
            canvas.style.display = 'none';
            
            // Add to document
            document.body.appendChild(video);
            document.body.appendChild(canvas);
            
            try {
                // Request camera access
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    }
                });
                
                // Set up video
                video.srcObject = stream;
                
                // Wait for video to be ready
                await new Promise((resolve) => {
                    video.onloadedmetadata = async () => {
                        try {
                            await video.play();
                            canvas.width = video.videoWidth;
                            canvas.height = video.videoHeight;
                            resolve();
                        } catch (err) {
                            console.error('Error playing video:', err);
                            throw err;
                        }
                    };
                });
                
                return true;
            } catch (err) {
                console.error('Error initializing webcam:', err);
                cleanup();
                return false;
            }
        }
        
        // Initialize immediately
        await initializeWebcam();
        """
        
        try:
            display(Javascript(js_code))
            time.sleep(3)  # Give time for initialization
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to start webcam: {str(e)}")
            return False
            
    def read_frame(self) -> Optional[np.ndarray]:
        """Capture and return a frame from the webcam."""
        if not self.is_initialized:
            return None
            
        js_code = """
        const video = document.querySelector('#captureVideo');
        const canvas = document.querySelector('#captureCanvas');
        
        if (!video || !canvas) {
            return null;
        }
        
        try {
            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0);
            return canvas.toDataURL('image/jpeg', 0.8);
        } catch (err) {
            console.error('Error capturing frame:', err);
            return null;
        }
        """
        
        try:
            frame_data = eval_js(js_code)
            if frame_data is None or not isinstance(frame_data, str):
                return None
                
            # Extract the base64 encoded image data
            image_data = frame_data.split(',')[1]
            image_bytes = b64decode(image_data)
            
            # Convert to numpy array and decode
            jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(jpg_as_np, flags=1)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error reading frame: {str(e)}")
            return None

    def stop(self) -> bool:
        """Stop the webcam capture and clean up resources."""
        if not self.is_initialized:
            return True
            
        js_code = """
        try {
            const video = document.querySelector('#captureVideo');
            const canvas = document.querySelector('#captureCanvas');
            
            if (video) {
                const stream = video.srcObject;
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                }
                video.remove();
            }
            
            if (canvas) {
                canvas.remove();
            }
            
            return true;
        } catch (err) {
            console.error('Error stopping webcam:', err);
            return false;
        }
        """
        
        try:
            result = eval_js(js_code)
            self.is_initialized = False
            return bool(result)
        except Exception as e:
            logger.error(f"Error stopping webcam: {str(e)}")
            self.is_initialized = False
            return False

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    def __init__(self):
        # JavaScript code is now wrapped in an IIFE (Immediately Invoked Function Expression)
        # to ensure proper initialization and scoping
        self.js_code = """
        (function() {
            async function createElements() {
                // Remove any existing elements
                if (window.video) window.video.remove();
                if (window.canvas) window.canvas.remove();
                
                // Create new elements
                window.video = document.createElement('video');
                window.canvas = document.createElement('caanvas');
                
                // Configure video element
                window.video.style.display = 'none';
                window.video.autoplay = true;
                window.video.playsinline = true;
                
                // Configure canvas
                window.canvas.style.display = 'none';
                
                // Add elements to document
                document.body.appendChild(window.video);
                document.body.appendChild(window.canvas);
            }

            window.startWebcam = async function() {
                await createElements();
                
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { 
                            facingMode: 'environment',
                            width: { ideal: 1280 },
                            height: { ideal: 720 }
                        } 
                    });
                    
                    window.video.srcObject = stream;
                    
                    // Wait for video to start playing
                    await new Promise((resolve) => {
                        window.video.onplaying = resolve;
                        window.video.play();
                    });
                    
                    // Set canvas dimensions after video starts
                    window.canvas.width = window.video.videoWidth;
                    window.canvas.height = window.video.videoHeight;
                    
                    return true;
                } catch(err) {
                    console.error("Error starting webcam:", err);
                    return false;
                }
            };
            
            window.captureFrame = function() {
                if (!window.video || !window.canvas) {
                    console.error("Video or canvas elements not initialized");
                    return null;
                }
                
                try {
                    const ctx = window.canvas.getContext('2d');
                    ctx.drawImage(window.video, 0, 0);
                    return window.canvas.toDataURL('image/jpeg', 0.8);
                } catch(err) {
                    console.error("Error capturing frame:", err);
                    return null;
                }
            };
            
            window.stopWebcam = function() {
                try {
                    if (window.video && window.video.srcObject) {
                        const stream = window.video.srcObject;
                        const tracks = stream.getTracks();
                        tracks.forEach(track => track.stop());
                    }
                    
                    if (window.video) window.video.remove();
                    if (window.canvas) window.canvas.remove();
                    
                    window.video = null;
                    window.canvas = null;
                    
                    return true;
                } catch(err) {
                    console.error("Error stopping webcam:", err);
                    return false;
                }
            };
        })();  // IIFE ends here
        """
        
    def start(self) -> bool:
        """Start the webcam capture. Returns True if successful."""
        try:
            display(Javascript(self.js_code))
            time.sleep(1)  # Give time for JS to initialize
            success = eval_js('startWebcam()')
            if not success:
                raise RuntimeError("Failed to start webcam")
            time.sleep(2)  # Give additional time for webcam to stabilize
            return True
        except Exception as e:
            print(f"Error starting webcam: {str(e)}")
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from the webcam. Returns None if unsuccessful."""
        try:
            # Verify elements exist before capturing
            elements_check = eval_js('!!(window.video && window.canvas)')
            if not elements_check:
                raise RuntimeError("Video or canvas elements not found")
            
            frame_data = eval_js('captureFrame()')
            if frame_data is None or not isinstance(frame_data, str):
                return None
                
            # Convert JS data to OpenCV image
            image_bytes = b64decode(frame_data.split(',')[1])
            jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(jpg_as_np, flags=1)
            
            return frame
        except Exception as e:
            print(f"Error reading frame: {str(e)}")
            return None
        
    def stop(self) -> bool:
        """Stop the webcam capture. Returns True if successful."""
        try:
            success = eval_js('stopWebcam()')
            if not success:
                raise RuntimeError("Failed to stop webcam")
            return True
        except Exception as e:
            print(f"Error stopping webcam: {str(e)}")
            return False