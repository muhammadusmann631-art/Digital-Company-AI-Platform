"""
Sentiment Analysis Module
Analyzes sentiment of user queries and generates contextual responses
"""
from transformers import pipeline
import streamlit as st
from langchain_groq import ChatGroq
import config


class SentimentAnalyzer:
    """Sentiment analysis and response generation"""
    
    def __init__(self):
        """Initialize sentiment analyzer and AI model"""
        # Load sentiment analysis pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=config.SENTIMENT_MODEL
        )
        
        # Initialize ChatGroq for responses
        self.ai_model = ChatGroq(
            model='llama-3.1-8b-instant',
            api_key=config.GROQ_API_KEY,
            temperature=0.8
        )
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text
        Args:
            text: Input text to analyze
        Returns:
            dict with sentiment label and score
        """
        try:
            result = self.sentiment_pipeline(text)[0]
            
            # Map labels to positive/negative/neutral
            label = result['label'].upper()
            score = result['score']
            
            # Determine sentiment
            if label == 'POSITIVE':
                sentiment = 'positive'
            elif label == 'NEGATIVE':
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'confidence': score,
                'label': label
            }
        except Exception as e:
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'label': 'NEUTRAL',
                'error': str(e)
            }
    
    def generate_response(self, query, sentiment):
        """
        Generate AI response based on query and sentiment
        Args:
            query: User's query
            sentiment: Detected sentiment (positive/negative/neutral)
        Returns:
            AI-generated response
        """
        # Create sentiment-aware prompt
        if sentiment == 'positive':
            system_prompt = (
                "The user seems happy and positive. Respond in an enthusiastic, "
                "encouraging, and uplifting manner. Match their positive energy!"
            )
        elif sentiment == 'negative':
            system_prompt = (
                "The user seems upset or negative. Respond with empathy, "
                "understanding, and support. Try to help them feel better."
            )
        else:
            system_prompt = (
                "The user has a neutral tone. Respond in a balanced, "
                "informative, and helpful manner."
            )
        
        # Create full prompt
        full_prompt = f"{system_prompt}\n\nUser query: {query}\n\nProvide a helpful response:"
        
        try:
            response = self.ai_model.invoke(full_prompt)
            return response.content
        except Exception as e:
            return f"I understand your query, but I'm having trouble generating a response right now. Error: {str(e)}"
    
    def get_sentiment_emoji(self, sentiment):
        """Get emoji for sentiment"""
        emoji_map = {
            'positive': '😊',
            'negative': '😔',
            'neutral': '😐'
        }
        return emoji_map.get(sentiment, '🤔')
    
    def get_sentiment_color(self, sentiment):
        """Get color for sentiment"""
        color_map = {
            'positive': '#10b981',  # Green
            'negative': '#ef4444',  # Red
            'neutral': '#f59e0b'    # Orange
        }
        return color_map.get(sentiment, '#6b7280')
