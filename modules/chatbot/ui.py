"""
Chatbot UI Module
Beautiful and colorful chatbot interface
"""
import streamlit as st
from .groq_client import ChatbotClient
import time


def render_chatbot_page():
    """Render the AI Chatbot page"""
    
    st.title("💬 AI Chatbot")
    st.markdown("---")
    
    # Initialize chatbot in session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ChatbotClient()
        # Add system message
        st.session_state.chatbot.add_system_message(
            "You are a helpful, friendly, and knowledgeable AI assistant. "
            "Provide clear, accurate, and engaging responses to user queries."
        )
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Chat Settings")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chatbot.clear_history()
            st.session_state.messages = []
            st.session_state.chatbot.add_system_message(
                "You are a helpful, friendly, and knowledgeable AI assistant. "
                "Provide clear, accurate, and engaging responses to user queries."
            )
            st.rerun()
        
        st.markdown("---")
        
        # Stats
        st.subheader("📊 Chat Statistics")
        st.metric("Messages", len(st.session_state.messages))
        
        st.markdown("---")
        
        # Export conversation
        if len(st.session_state.messages) > 0:
            if st.button("📥 Export Chat", use_container_width=True):
                conversation = st.session_state.chatbot.export_conversation()
                st.download_button(
                    label="Download Conversation",
                    data=conversation,
                    file_name="chat_conversation.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        st.markdown("---")
        
        # About
        st.subheader("ℹ️ About")
        st.markdown("""
        This chatbot is powered by:
        - 🤖 **ChatGroq**
        - 🧠 **Llama 3.1 8B**
        
        Features:
        - 💬 Natural conversations
        - 🧠 Context awareness
        - ⚡ Fast responses
        - 📝 Export chat history
        """)
    
    # Main chat area
    st.header("💭 Conversation")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(f"**You:** {message['content']}")
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(f"**AI:** {message['content']}")
    
    # Chat input
    user_input = st.chat_input("Type your message here...", key="chat_input")
    
    if user_input:
        # Add user message to display
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message immediately
        with chat_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"**You:** {user_input}")
        
        # Get AI response
        with st.spinner("🤔 Thinking..."):
            ai_response = st.session_state.chatbot.get_response(user_input)
        
        # Add AI response to display
        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_response
        })
        
        # Display AI response
        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"**AI:** {ai_response}")
        
        # Rerun to update the display
        st.rerun()
    
    # Welcome message if no messages
    if len(st.session_state.messages) == 0:
        st.info("👋 Welcome! Ask me anything and I'll do my best to help you!")
        
        # Suggested questions
        st.markdown("### 💡 Suggested Questions:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🌍 Tell me about AI", key="q1"):
                user_input = "Tell me about artificial intelligence"
                st.session_state.messages.append({"role": "user", "content": user_input})
                ai_response = st.session_state.chatbot.get_response(user_input)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                st.rerun()
            
            if st.button("💻 Explain machine learning", key="q2"):
                user_input = "Explain machine learning in simple terms"
                st.session_state.messages.append({"role": "user", "content": user_input})
                ai_response = st.session_state.chatbot.get_response(user_input)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                st.rerun()
        
        with col2:
            if st.button("🎨 What is computer vision?", key="q3"):
                user_input = "What is computer vision and how does it work?"
                st.session_state.messages.append({"role": "user", "content": user_input})
                ai_response = st.session_state.chatbot.get_response(user_input)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                st.rerun()
            
            if st.button("📊 Explain data analytics", key="q4"):
                user_input = "What is data analytics and why is it important?"
                st.session_state.messages.append({"role": "user", "content": user_input})
                ai_response = st.session_state.chatbot.get_response(user_input)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                st.rerun()
