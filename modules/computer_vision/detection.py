"""
Object Detection Module
Handles object detection inference and visualization
"""
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image


class ObjectDetector:
    """Object detection using YOLO"""
    
    def __init__(self, model_path="yolo11n.pt"):
        self.model = YOLO(model_path)
    
    def detect(self, image, conf_threshold=0.25):
        """
        Detect objects in image
        Args:
            image: PIL Image or numpy array
            conf_threshold: Confidence threshold
        Returns:
            results: Detection results
        """
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            verbose=False
        )
        return results
    
    def draw_detections(self, image, results):
        """Draw bounding boxes on image"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        annotated_image = results[0].plot()
        return annotated_image
    
    def get_detection_info(self, results):
        """Extract detection information"""
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    'class': result.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                }
                detections.append(detection)
        
        return detections
