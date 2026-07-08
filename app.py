"""
AI Multi-Tool Platform
Main Streamlit Application

A comprehensive AI platform with 5 powerful modules:
1. Data Analytics & Preprocessing
2. Computer Vision with YOLO11n
3. AI Chatbot
4. Sentiment Analysis
5. Emotion-Based Poetry
"""
import streamlit as st
import config
from pathlib import Path

# Import module UIs
from modules.data_analytics.ui import render_data_analytics_page
from modules.computer_vision.ui import render_computer_vision_page
from modules.chatbot.ui import render_chatbot_page
from modules.sentiment.ui import render_sentiment_page

# Try to import emotion poetry, make it optional
try:
    from modules.emotion_poetry.ui import render_emotion_poetry_page
    EMOTION_POETRY_AVAILABLE = True
except ImportError as e:
    EMOTION_POETRY_AVAILABLE = False
    print(f"Warning: Emotion Poetry module not available: {e}")


# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load custom CSS
def load_css():
    """Load custom CSS styling"""
    css_file = Path(__file__).parent / "assets" / "styles.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Load CSS
load_css()


# Sidebar navigation
def render_sidebar():
    """Render sidebar navigation"""
    with st.sidebar:
        st.title(config.APP_TITLE)
        st.markdown("---")
        
        # Navigation
        st.header("🧭 Navigation")
        
        # Build navigation options
        nav_options = [
            "🏠 Home",
            "📊 Data Analytics",
            "🎯 Computer Vision",
            "💬 AI Chatbot",
            "😊 Sentiment Analysis"
        ]
        
        if EMOTION_POETRY_AVAILABLE:
            nav_options.append("📝 Emotion Poetry")
        
        page = st.radio(
            "Select Module",
            nav_options,
            key="navigation"
        )
        
        st.markdown("---")
        
        # About section
        st.header("ℹ️ About")
        st.markdown("""
        **AI Multi-Tool Platform** is a comprehensive suite of AI-powered tools for:
        
        - 📊 Data preprocessing & analytics
        - 🎯 Computer vision tasks
        - 💬 Intelligent conversations
        - 😊 Sentiment analysis
        - 📝 Creative poetry generation
        
        **Version:** 1.0.0  
        **Powered by:** Streamlit, YOLO11n, ChatGroq
        """)
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style='text-align: center; color: #888;'>
            <p>Made with ❤️ using AI</p>
        </div>
        """, unsafe_allow_html=True)
        
        return page


# Home page
def render_home_page():
    """Render home page"""
    st.title("🚀 Welcome to AI Multi-Tool Platform")
    st.markdown("---")
    
    st.markdown("""
    ### Your All-in-One AI Solution
    
    This platform combines **5 powerful AI modules** to help you with various tasks:
    """)
    
    # Feature cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📊 Data Analytics
        - Upload and analyze CSV data
        - Apply preprocessing methods
        - One-Hot & Label Encoding
        - TF-IDF Vectorization
        - Standard, MinMax, Robust Scaling
        - Interactive dashboards
        - Download processed data
        """)
        
        st.markdown("""
        #### 🎯 Computer Vision
        - Object Detection
        - Instance Segmentation
        - Oriented Bounding Boxes
        - Pose Estimation
        - YOLO11n fine-tuning
        - Real-time inference
        """)
        
        st.markdown("""
        #### 💬 AI Chatbot
        - Powered by ChatGroq
        - Natural conversations
        - Context awareness
        - Chat history
        - Export conversations
        """)
    
    with col2:
        st.markdown("""
        #### 😊 Sentiment Analysis
        - Detect sentiment (positive/negative/neutral)
        - AI-powered responses
        - Context-aware replies
        - Analysis history
        - Confidence scores
        """)
        
        st.markdown("""
        #### 📝 Emotion Poetry
        - Real-time emotion detection
        - Camera-based face analysis
        - AI poetry generation
        - Multiple poetry styles
        - Emotion-specific poems
        - Poetry collection
        """)
    
    st.markdown("---")
    
    # Getting started
    st.header("🚀 Getting Started")
    
    st.markdown("""
    1. **Select a module** from the sidebar navigation
    2. **Follow the instructions** on each page
    3. **Upload data** or **interact** with the AI tools
    4. **Enjoy** the powerful AI capabilities!
    """)
    
    st.markdown("---")
    
    # Quick stats
    st.header("📊 Platform Features")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Modules", "5", help="Total number of AI modules")
    
    with col2:
        st.metric("AI Models", "3+", help="ChatGroq, YOLO11n, Transformers")
    
    with col3:
        st.metric("Preprocessing Methods", "6+", help="Encoding, scaling, vectorization")
    
    with col4:
        st.metric("CV Tasks", "4", help="Detection, segmentation, OBB, pose")
    
    st.markdown("---")
    
    # Tips
    st.header("💡 Tips")
    
    st.info("""
    **Pro Tips:**
    - Start with Data Analytics to preprocess your datasets
    - Use Computer Vision for image/video analysis
    - Chat with the AI for quick answers
    - Analyze sentiment of customer feedback
    - Generate creative poetry based on emotions
    """)
    
    st.success("""
    **Ready to explore?** Select a module from the sidebar to get started! 🎉
    """)


# Main app
def main():
    """Main application"""
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render selected page
    if page == "🏠 Home":
        render_home_page()
    elif page == "📊 Data Analytics":
        render_data_analytics_page()
    elif page == "🎯 Computer Vision":
        render_computer_vision_page()
    elif page == "💬 AI Chatbot":
        render_chatbot_page()
    elif page == "😊 Sentiment Analysis":
        render_sentiment_page()
    elif page == "📝 Emotion Poetry":
        if EMOTION_POETRY_AVAILABLE:
            render_emotion_poetry_page()
        else:
            st.error("❌ Emotion Poetry module is not available due to missing dependencies (fer package).")
            st.info("To enable this module, install the fer package: `pip install fer`")


if __name__ == "__main__":
    main()
