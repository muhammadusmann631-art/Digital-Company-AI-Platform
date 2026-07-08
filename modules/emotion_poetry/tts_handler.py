"""
Text-to-Speech Handler Module
Generates audio for English and Urdu poems
"""
import os
import base64
from gtts import gTTS
from pathlib import Path
import tempfile
import hashlib


class TTSHandler:
    """Handle text-to-speech for bilingual poetry"""
    
    def __init__(self):
        """Initialize TTS handler"""
        self.temp_dir = Path(tempfile.gettempdir()) / "emotion_poetry_audio"
        self.temp_dir.mkdir(exist_ok=True)
        
    def _get_cache_path(self, text, lang):
        """Generate cache file path based on text hash"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.temp_dir / f"{lang}_{text_hash}.mp3"
    
    def generate_english_speech(self, text):
        """
        Generate English speech from text
        Args:
            text: English poem text
        Returns:
            base64 encoded audio string
        """
        try:
            cache_path = self._get_cache_path(text, 'en')
            
            # Check if cached
            if not cache_path.exists():
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(str(cache_path))
            
            return self._get_audio_base64(cache_path)
            
        except Exception as e:
            print(f"Error generating English speech: {e}")
            return None
    
    def generate_urdu_speech(self, text):
        """
        Generate Urdu speech from text
        Args:
            text: Urdu poem text
        Returns:
            base64 encoded audio string
        """
        try:
            cache_path = self._get_cache_path(text, 'ur')
            
            # Check if cached
            if not cache_path.exists():
                tts = gTTS(text=text, lang='ur', slow=False)
                tts.save(str(cache_path))
            
            return self._get_audio_base64(cache_path)
            
        except Exception as e:
            print(f"Error generating Urdu speech: {e}")
            return None
    
    def _get_audio_base64(self, audio_path):
        """
        Convert audio file to base64 for HTML5 player
        Args:
            audio_path: Path to audio file
        Returns:
            base64 encoded string
        """
        try:
            with open(audio_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode()
                return audio_base64
        except Exception as e:
            print(f"Error encoding audio: {e}")
            return None
    
    def cleanup_old_files(self, max_files=50):
        """
        Clean up old audio files to save space
        Args:
            max_files: Maximum number of files to keep
        """
        try:
            files = sorted(self.temp_dir.glob("*.mp3"), key=os.path.getmtime)
            if len(files) > max_files:
                for file in files[:-max_files]:
                    file.unlink()
        except Exception as e:
            print(f"Error cleaning up files: {e}")
