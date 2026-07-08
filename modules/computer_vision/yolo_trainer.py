"""
YOLO Training Module
Handles YOLO11n model training for detection, segmentation, OBB, and pose
"""
import os
from ultralytics import YOLO
import streamlit as st
from pathlib import Path


class YOLOTrainer:
    """YOLO model training and fine-tuning"""
    
    def __init__(self, task_type="detection"):
        """
        Initialize YOLO trainer
        Args:
            task_type: 'detection', 'segmentation', 'obb', or 'pose'
        """
        self.task_type = task_type
        self.model = None
        self.model_map = {
            "detection": "yolo11n.pt",
            "segmentation": "yolo11n-seg.pt",
            "obb": "yolo11n-obb.pt",
            "pose": "yolo11n-pose.pt"
        }
    
    def load_model(self):
        """Load pretrained YOLO model"""
        model_name = self.model_map.get(self.task_type, "yolo11n.pt")
        self.model = YOLO(model_name)
        return self.model
    
    def train(self, data_yaml, epochs=50, imgsz=640, batch=16, device='cpu'):
        """
        Train YOLO model
        Args:
            data_yaml: Path to data.yaml file
            epochs: Number of training epochs
            imgsz: Image size
            batch: Batch size
            device: 'cpu' or 'cuda'
        """
        if self.model is None:
            self.load_model()
        
        # Train the model
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            patience=10,
            save=True,
            plots=True,
            verbose=True
        )
        
        return results
    
    def validate(self):
        """Validate the trained model"""
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        metrics = self.model.val()
        return metrics
    
    def predict(self, source, conf=0.25, save=True):
        """
        Run inference on images/videos
        Args:
            source: Path to image/video or directory
            conf: Confidence threshold
            save: Save results
        """
        if self.model is None:
            self.load_model()
        
        results = self.model.predict(
            source=source,
            conf=conf,
            save=save,
            show_labels=True,
            show_conf=True
        )
        
        return results
    
    def export_model(self, format='onnx'):
        """Export model to different formats"""
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        self.model.export(format=format)
    
    def get_model_info(self):
        """Get model information"""
        if self.model is None:
            self.load_model()
        
        info = {
            'task': self.task_type,
            'model_name': self.model_map[self.task_type],
            'parameters': sum(p.numel() for p in self.model.model.parameters()),
        }
        return info


def create_data_yaml(train_path, val_path, class_names, save_path):
    """
    Create data.yaml file for YOLO training
    Args:
        train_path: Path to training images
        val_path: Path to validation images
        class_names: List of class names
        save_path: Where to save data.yaml
    """
    yaml_content = f"""# Dataset configuration for YOLO training
path: {os.path.dirname(train_path)}
train: {os.path.basename(train_path)}
val: {os.path.basename(val_path)}

# Classes
nc: {len(class_names)}
names: {class_names}
"""
    
    with open(save_path, 'w') as f:
        f.write(yaml_content)
    
    return save_path
