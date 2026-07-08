"""
Configuration file for AI Multi-Tool Platform
"""
import os

# API Keys
GROQ_API_KEY = ""

# Model Paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
YOLO_MODELS = {
    "detection": "yolo11n.pt",
    "segmentation": "yolo11n-seg.pt",
    "obb": "yolo11n-obb.pt",
    "pose": "yolo11n-pose.pt"
}

# Sentiment Model
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Emotion Detection
EMOTION_MODEL = "deepface"  # or "fer"

# App Settings
APP_TITLE = "🚀 AI Multi-Tool Platform"
PAGE_ICON = "🤖"

# Color Scheme
COLORS = {
    "primary": "#6366f1",
    "secondary": "#8b5cf6",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "info": "#3b82f6",
    "dark": "#1f2937",
    "light": "#f9fafb"
}
