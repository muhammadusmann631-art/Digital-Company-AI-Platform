"""
Emotion Detection Module
Detects emotions from facial expressions using camera
"""
import cv2
import numpy as np
from fer.fer import FER
import streamlit as st


class EmotionDetector:
    """Real-time emotion detection from camera"""
    
    def __init__(self):
        """Initialize emotion detector"""
        try:
            # Try with MTCNN but have it as a property
            self.detector_mtcnn = FER(mtcnn=True)
            self.detector_simple = FER(mtcnn=False)
            self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        except Exception as e:
            st.error(f"Error initializing emotion detector: {str(e)}")
            self.detector_mtcnn = None
            self.detector_simple = None
    
    def detect_emotion_from_frame(self, frame):
        """
        Detect emotion from a single frame
        Args:
            frame: OpenCV frame (BGR format)
        Returns:
            dict with dominant emotion and all scores
        """
        if self.detector_mtcnn is None:
            return None
        
        try:
            # Try with MTCNN first
            result = self.detector_mtcnn.detect_emotions(frame)
            
            # If MTCNN fails, try simple detector
            if not result or len(result) == 0:
                result = self.detector_simple.detect_emotions(frame)
            
            if result and len(result) > 0:
                # Get the first face detected
                emotions = result[0]['emotions']
                
                # Find dominant emotion
                dominant_emotion = max(emotions, key=emotions.get)
                confidence = emotions[dominant_emotion]
                
                return {
                    'dominant_emotion': dominant_emotion,
                    'confidence': confidence,
                    'all_emotions': emotions,
                    'face_box': result[0]['box']
                }
            else:
                return None
                
        except Exception as e:
            print(f"Detection error: {e}")
            return None
    
    def get_emotion_emoji(self, emotion):
        """Get emoji for emotion"""
        emoji_map = {
            'angry': '😠',
            'disgust': '🤢',
            'fear': '😨',
            'happy': '😊',
            'sad': '😢',
            'surprise': '😲',
            'neutral': '😐'
        }
        return emoji_map.get(emotion, '🤔')
    
    def get_emotion_color(self, emotion):
        """Get color for emotion"""
        color_map = {
            'angry': '#ef4444',      # Red
            'disgust': '#84cc16',    # Lime
            'fear': '#8b5cf6',       # Purple
            'happy': '#10b981',      # Green
            'sad': '#3b82f6',        # Blue
            'surprise': '#f59e0b',   # Orange
            'neutral': '#6b7280'     # Gray
        }
        return color_map.get(emotion, '#6b7280')
    
    def draw_emotion_on_frame(self, frame, emotion_result):
        """Draw emotion detection results on frame"""
        if emotion_result is None:
            return frame
        
        # Get face box
        box = emotion_result['face_box']
        x, y, w, h = box
        
        # Draw rectangle around face
        color = self.get_emotion_color(emotion_result['dominant_emotion'])
        # Convert hex to BGR
        color_bgr = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, 3)
        
        # Draw emotion label
        emotion_text = f"{emotion_result['dominant_emotion'].upper()} ({emotion_result['confidence']:.2f})"
        cv2.putText(frame, emotion_text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2)
        
        return frame
    
    def capture_from_camera(self):
        """Capture frame from camera"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            return None, "Cannot access camera"
        
        # Skip first few frames to allow camera to adjust exposure
        for _ in range(10):
            cap.read()
            
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame, None
        else:
            return None, "Failed to capture frame"
