"""
Segmentation Module
Handles instance segmentation using YOLO
"""
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image


class InstanceSegmentation:
    """Instance segmentation using YOLO"""
    
    def __init__(self, model_path="yolo11n-seg.pt"):
        self.model = YOLO(model_path)
    
    def segment(self, image, conf_threshold=0.25):
        """
        Segment objects in image
        Args:
            image: PIL Image or numpy array
            conf_threshold: Confidence threshold
        Returns:
            results: Segmentation results
        """
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            verbose=False
        )
        return results
    
    def draw_segmentation(self, image, results):
        """Draw segmentation masks on image"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        annotated_image = results[0].plot()
        return annotated_image
    
    def get_segmentation_info(self, results):
        """Extract segmentation information"""
        segments = []
        
        for result in results:
            if result.masks is not None:
                boxes = result.boxes
                masks = result.masks
                
                for i, (box, mask) in enumerate(zip(boxes, masks)):
                    segment = {
                        'class': result.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': box.xyxy[0].tolist(),
                        'mask_shape': mask.data.shape
                    }
                    segments.append(segment)
        
        return segments
