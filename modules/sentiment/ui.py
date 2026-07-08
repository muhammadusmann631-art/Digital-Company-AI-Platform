"""
Sentiment Analysis UI Module
Interface for sentiment analysis and AI responses
"""
import streamlit as st
from .analyzer import SentimentAnalyzer
import time


def render_sentiment_page():
    """Render the Sentiment Analysis page"""
    
    st.title("😊 Sentiment Analysis")
    st.markdown("---")
    
    # Initialize analyzer
    if 'sentiment_analyzer' not in st.session_state:
        with st.spinner("Loading sentiment analyzer..."):
            st.session_state.sentiment_analyzer = SentimentAnalyzer()
    
    if 'sentiment_history' not in st.session_state:
        st.session_state.sentiment_history = []
    
    # Main content
    st.header("📝 Analyze Your Query")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input
        user_query = st.text_area(
            "Enter your text or query",
            placeholder="Type something here... I'll analyze the sentiment and provide a response!",
            height=150,
            key="sentiment_input"
        )
        
        # Analyze button
        analyze_button = st.button(
            "🔍 Analyze Sentiment",
            use_container_width=True,
            type="primary"
        )
    
    with col2:
        st.subheader("📊 Quick Stats")
        
        if len(st.session_state.sentiment_history) > 0:
            positive_count = sum(1 for h in st.session_state.sentiment_history if h['sentiment'] == 'positive')
            negative_count = sum(1 for h in st.session_state.sentiment_history if h['sentiment'] == 'negative')
            neutral_count = sum(1 for h in st.session_state.sentiment_history if h['sentiment'] == 'neutral')
            
            st.metric("😊 Positive", positive_count)
            st.metric("😔 Negative", negative_count)
            st.metric("😐 Neutral", neutral_count)
        else:
            st.info("No analysis yet")
    
    st.markdown("---")
    
    # Process query
    if analyze_button and user_query:
        with st.spinner("Analyzing sentiment..."):
            # Analyze sentiment
            sentiment_result = st.session_state.sentiment_analyzer.analyze_sentiment(user_query)
            
            # Get emoji and color
            emoji = st.session_state.sentiment_analyzer.get_sentiment_emoji(sentiment_result['sentiment'])
            color = st.session_state.sentiment_analyzer.get_sentiment_color(sentiment_result['sentiment'])
            
            # Display sentiment result
            st.subheader("🎯 Sentiment Analysis Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"### {emoji} Sentiment")
                st.markdown(f"<h2 style='color: {color};'>{sentiment_result['sentiment'].upper()}</h2>", 
                           unsafe_allow_html=True)
            
            with col2:
                st.metric(
                    "Confidence",
                    f"{sentiment_result['confidence']:.1%}"
                )
            
            with col3:
                st.metric(
                    "Label",
                    sentiment_result['label']
                )
            
            st.markdown("---")
            
            # Generate AI response
            with st.spinner("Generating AI response..."):
                ai_response = st.session_state.sentiment_analyzer.generate_response(
                    user_query,
                    sentiment_result['sentiment']
                )
            
            # Display AI response
            st.subheader("🤖 AI Response")
            
            # Create styled response box
            response_style = f"""
            <div style='
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
                padding: 1.5rem;
                border-radius: 15px;
                border-left: 5px solid {color};
                margin: 1rem 0;
            '>
                <p style='font-size: 1.1rem; line-height: 1.6; margin: 0;'>{ai_response}</p>
            </div>
            """
            st.markdown(response_style, unsafe_allow_html=True)
            
            # Add to history
            st.session_state.sentiment_history.append({
                'query': user_query,
                'sentiment': sentiment_result['sentiment'],
                'confidence': sentiment_result['confidence'],
                'response': ai_response,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            st.success("✅ Analysis complete!")
    
    elif analyze_button and not user_query:
        st.warning("⚠️ Please enter some text to analyze!")
    
    # Show history
    if len(st.session_state.sentiment_history) > 0:
        st.markdown("---")
        st.header("📜 Analysis History")
        
        # Clear history button
        if st.button("🗑️ Clear History", key="clear_sentiment_history"):
            st.session_state.sentiment_history = []
            st.rerun()
        
        # Display history in reverse order (newest first)
        for idx, item in enumerate(reversed(st.session_state.sentiment_history)):
            emoji = st.session_state.sentiment_analyzer.get_sentiment_emoji(item['sentiment'])
            color = st.session_state.sentiment_analyzer.get_sentiment_color(item['sentiment'])
            
            with st.expander(f"{emoji} {item['sentiment'].title()} - {item['timestamp']}", expanded=(idx == 0)):
                st.markdown(f"**Query:** {item['query']}")
                st.markdown(f"**Sentiment:** <span style='color: {color}; font-weight: bold;'>{item['sentiment'].upper()}</span> ({item['confidence']:.1%} confidence)", unsafe_allow_html=True)
                st.markdown(f"**AI Response:** {item['response']}")
    
    else:
        # Show example
        st.markdown("---")
        st.subheader("✨ How It Works")
        
        st.markdown("""
        1. **Enter your text** - Type any query, message, or text you want to analyze
        2. **Analyze sentiment** - Our AI will detect if it's positive, negative, or neutral
        3. **Get AI response** - Receive a contextual response based on the detected sentiment
        
        **Examples to try:**
        - "I'm so happy today! Everything is going great!" (Positive)
        - "I'm feeling frustrated with this problem." (Negative)
        - "What is the weather like?" (Neutral)
        """)
        
        st.markdown("---")
        
        # Quick examples
        st.subheader("🚀 Quick Examples")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("😊 Positive Example", use_container_width=True):
                st.session_state.sentiment_input = "I absolutely love this! It's amazing and wonderful!"
                st.rerun()
        
        with col2:
            if st.button("😔 Negative Example", use_container_width=True):
                st.session_state.sentiment_input = "This is terrible and frustrating. I'm very disappointed."
                st.rerun()
        
        with col3:
            if st.button("😐 Neutral Example", use_container_width=True):
                st.session_state.sentiment_input = "Can you explain how this works?"
                st.rerun()
