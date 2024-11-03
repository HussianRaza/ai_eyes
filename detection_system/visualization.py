

# detection_system/visualization.py
import cv2
import numpy as np
from typing import Tuple, Optional

from detection_system.models import Detection, FrameResults

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
