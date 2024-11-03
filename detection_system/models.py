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