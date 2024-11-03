
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