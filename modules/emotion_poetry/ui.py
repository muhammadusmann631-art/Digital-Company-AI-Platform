"""
Emotion Poetry UI Module
Interface for emotion detection and bilingual poetry generation
"""
import streamlit as st
import cv2
from PIL import Image
import numpy as np
from .emotion_detector import EmotionDetector
from .poetry_generator import PoetryGenerator
from .tts_handler import TTSHandler
import time


def render_emotion_poetry_page():
    """Render the Emotion Poetry page"""
    
    st.title("📝 Emotion-Based Poetry Generator")
    st.markdown("---")
    
    # Initialize components
    if 'emotion_detector' not in st.session_state:
        with st.spinner("Loading emotion detector..."):
            st.session_state.emotion_detector = EmotionDetector()
    
    if 'poetry_generator' not in st.session_state:
        st.session_state.poetry_generator = PoetryGenerator()
    
    if 'tts_handler' not in st.session_state:
        st.session_state.tts_handler = TTSHandler()
    
    if 'poetry_history' not in st.session_state:
        st.session_state.poetry_history = []
    
    # Main content
    st.header("📸 Capture Your Emotion")
    
    st.info("👇 Click the button below to capture your face and detect your emotion. AI will generate a beautiful bilingual poem (English + Urdu) based on your emotion!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Capture button
        capture_button = st.button(
            "📸 Capture & Generate Bilingual Poetry",
            use_container_width=True,
            type="primary"
        )
    
    with col2:
        st.subheader("📊 Emotion Stats")
        if len(st.session_state.poetry_history) > 0:
            emotions = [p['emotion'] for p in st.session_state.poetry_history]
            most_common = max(set(emotions), key=emotions.count)
            emoji = st.session_state.emotion_detector.get_emotion_emoji(most_common)
            st.metric("Most Common Emotion", f"{emoji} {most_common.title()}")
            st.metric("Total Poems", len(st.session_state.poetry_history))
        else:
            st.info("No poems generated yet")
    
    st.markdown("---")
    
    # Process capture
    if capture_button:
        with st.spinner("📸 Accessing camera..."):
            # Capture from camera
            frame, error = st.session_state.emotion_detector.capture_from_camera()
            
            if error:
                st.error(f"❌ {error}")
                st.info("💡 Make sure your camera is connected and not being used by another application.")
            else:
                # Detect emotion
                with st.spinner("🔍 Detecting emotion..."):
                    emotion_result = st.session_state.emotion_detector.detect_emotion_from_frame(frame)
                
                if emotion_result is None:
                    st.warning("⚠️ No face detected! Please make sure your face is clearly visible.")
                else:
                    # Draw emotion on frame
                    annotated_frame = st.session_state.emotion_detector.draw_emotion_on_frame(
                        frame.copy(), 
                        emotion_result
                    )
                    
                    # Display results
                    st.subheader("🎭 Detected Emotion")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Show captured image
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        st.image(rgb_frame, caption="Captured Image", use_column_width=True)
                    
                    with col2:
                        # Show emotion details
                        emotion = emotion_result['dominant_emotion']
                        emoji = st.session_state.emotion_detector.get_emotion_emoji(emotion)
                        color = st.session_state.emotion_detector.get_emotion_color(emotion)
                        
                        st.markdown(f"### {emoji} {emotion.upper()}")
                        st.markdown(f"<h3 style='color: {color};'>Confidence: {emotion_result['confidence']:.1%}</h3>", 
                                   unsafe_allow_html=True)
                        
                        # Show all emotions
                        st.markdown("**All Detected Emotions:**")
                        for emo, score in sorted(emotion_result['all_emotions'].items(), 
                                                key=lambda x: x[1], reverse=True):
                            st.progress(score, text=f"{emo.title()}: {score:.1%}")
                    
                    st.markdown("---")
                    
                    # Generate bilingual poetry
                    with st.spinner("✍️ Generating bilingual poetry..."):
                        description = st.session_state.poetry_generator.get_emotion_description(emotion)
                        st.info(description)
                        
                        poems = st.session_state.poetry_generator.generate_bilingual_poetry(emotion)
                    
                    # Display bilingual poetry
                    st.subheader("📜 Your Emotion Poem (English + اردو)")
                    
                    # Create bilingual display
                    display_bilingual_poetry(poems, emotion, color)
                    
                    # Generate text-to-speech
                    with st.spinner("🔊 Generating audio..."):
                        english_audio = st.session_state.tts_handler.generate_english_speech(poems['english'])
                        urdu_audio = st.session_state.tts_handler.generate_urdu_speech(poems['urdu'])
                    
                    # Add to history
                    st.session_state.poetry_history.append({
                        'emotion': emotion,
                        'confidence': emotion_result['confidence'],
                        'poems': poems,
                        'english_audio': english_audio,
                        'urdu_audio': urdu_audio,
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'image': rgb_frame
                    })
                    
                    st.success("✅ Bilingual poetry generated successfully!")
    
    # Show history
    if len(st.session_state.poetry_history) > 0:
        st.markdown("---")
        st.header("📚 Poetry Collection")
        
        # Clear history button
        if st.button("🗑️ Clear Collection", key="clear_poetry_history"):
            st.session_state.poetry_history = []
            st.rerun()
        
        # Display history
        for idx, item in enumerate(reversed(st.session_state.poetry_history)):
            emoji = st.session_state.emotion_detector.get_emotion_emoji(item['emotion'])
            color = st.session_state.emotion_detector.get_emotion_color(item['emotion'])
            
            with st.expander(
                f"{emoji} {item['emotion'].title()} - {item['timestamp']}", 
                expanded=(idx == 0)
            ):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(item['image'], caption="Captured Emotion", use_column_width=True)
                    st.markdown(f"**Emotion:** <span style='color: {color}; font-weight: bold;'>{item['emotion'].upper()}</span>", 
                               unsafe_allow_html=True)
                    st.markdown(f"**Confidence:** {item['confidence']:.1%}")
                
                with col2:
                    st.markdown("**Generated Bilingual Poem:**")
                    display_bilingual_poetry(item['poems'], item['emotion'], color, 
                                            item.get('english_audio'), item.get('urdu_audio'))
    
    else:
        # Show instructions
        st.markdown("---")
        st.subheader("✨ How It Works")
        
        st.markdown("""
        1. **📸 Capture** - Click the button to capture your face from the camera
        2. **🎭 Detect** - AI detects your emotion (angry, sad, happy, surprise, fear, disgust, neutral)
        3. **✍️ Generate** - AI creates beautiful poems in both English and Urdu (5 stanzas each)
        4. **🔊 Listen** - Play the poems with text-to-speech in both languages
        5. **📚 Save** - Your poems are saved in the collection below
        
        **Supported Emotions:**
        - 😠 Angry - Powerful, intense poetry
        - 😢 Sad - Melancholic, touching poetry
        - 😊 Happy - Joyful, uplifting poetry
        - 😨 Fear - Suspenseful, atmospheric poetry
        - 😲 Surprise - Exciting, dynamic poetry
        - 🤢 Disgust - Critical, sharp poetry
        - 😐 Neutral - Contemplative, balanced poetry
        """)
        
        st.markdown("---")
        
        st.info("💡 **Tip:** Make sure you're in a well-lit area and your face is clearly visible to the camera for best results!")


def display_bilingual_poetry(poems, emotion, color, english_audio=None, urdu_audio=None):
    """Display bilingual poetry with beautiful formatting and audio players"""
    
    # Load Google Fonts for Urdu
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap" rel="stylesheet">
    <style>
        .poem-box {
            padding: 2rem;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
            margin-bottom: 1rem;
            min-height: 400px;
        }
        
        .english-box {
            border-left: 5px solid """ + color + """;
        }
        
        .urdu-box {
            border-right: 5px solid """ + color + """;
            direction: rtl;
            text-align: right;
        }
        
        .poem-header {
            font-size: 1.3rem;
            font-weight: bold;
            margin-bottom: 1rem;
            color: """ + color + """;
        }
        
        .poem-content {
            font-style: italic;
            line-height: 1.8;
            white-space: pre-line;
            font-size: 1.1rem;
        }
        
        .urdu-content {
            font-family: 'Noto Nastaliq Urdu', serif;
            font-size: 1.4rem;
            line-height: 2.5;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create two columns for bilingual display
    col1, col2 = st.columns(2)
    
    with col1:
        # English Poem
        st.markdown(f"""
        <div class="poem-box english-box">
            <div class="poem-header">📖 English Poem</div>
            <div class="poem-content">{poems['english']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if english_audio:
            # Use st.audio for better reliability
            import base64
            audio_data = base64.b64decode(english_audio)
            st.audio(audio_data, format='audio/mp3')

    with col2:
        # Urdu Poem
        st.markdown(f"""
        <div class="poem-box urdu-box">
            <div class="poem-header">📖 اردو نظم</div>
            <div class="poem-content urdu-content">{poems['urdu']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if urdu_audio:
            # Use st.audio for better reliability
            import base64
            audio_data = base64.b64decode(urdu_audio)
            st.audio(audio_data, format='audio/mp3')
