"""
ChatGroq Client Module
Handles ChatGroq API integration for AI chatbot
"""
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import streamlit as st
import config


class ChatbotClient:
    """ChatGroq-powered chatbot"""
    
    def __init__(self):
        """Initialize ChatGroq client"""
        self.model = ChatGroq(
            model='llama-3.1-8b-instant',
            api_key=config.GROQ_API_KEY,
            temperature=0.7,
            max_tokens=1024
        )
        self.conversation_history = []
    
    def add_system_message(self, content):
        """Add system message to conversation"""
        self.conversation_history.append(SystemMessage(content=content))
    
    def add_user_message(self, content):
        """Add user message to conversation"""
        self.conversation_history.append(HumanMessage(content=content))
    
    def add_ai_message(self, content):
        """Add AI message to conversation"""
        self.conversation_history.append(AIMessage(content=content))
    
    def get_response(self, user_message):
        """
        Get AI response for user message
        Args:
            user_message: User's input message
        Returns:
            AI response text
        """
        # Add user message to history
        self.add_user_message(user_message)
        
        try:
            # Get response from ChatGroq
            response = self.model.invoke(self.conversation_history)
            ai_response = response.content
            
            # Add AI response to history
            self.add_ai_message(ai_response)
            
            return ai_response
            
        except Exception as e:
            error_msg = f"Error getting response: {str(e)}"
            return error_msg
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_history_length(self):
        """Get number of messages in history"""
        return len(self.conversation_history)
    
    def export_conversation(self):
        """Export conversation as text"""
        conversation_text = ""
        for msg in self.conversation_history:
            if isinstance(msg, HumanMessage):
                conversation_text += f"User: {msg.content}\n\n"
            elif isinstance(msg, AIMessage):
                conversation_text += f"AI: {msg.content}\n\n"
            elif isinstance(msg, SystemMessage):
                conversation_text += f"System: {msg.content}\n\n"
        return conversation_text
