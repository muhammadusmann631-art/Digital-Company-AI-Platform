<<<<<<< HEAD
# AI Multi-Tool Platform 🚀

A comprehensive, all-in-one AI platform featuring powerful modules for data analytics, computer vision, natural language processing, and creative arts. Built with **Streamlit**, powered by **State-of-the-Art AI models**.

---

## 🌟 Features & Modules

### 1. 📊 Data Analytics & Preprocessing
- **Interactive Dashboard**: Explore datasets with Plotly, Seaborn, and Matplotlib.
- **Robust Preprocessing**: Handle missing values, outliers, and feature scaling (Standard, MinMax, Robust).
- **Encoding & Vectorization**: Apply One-Hot, Label Encoding, and TF-IDF for machine learning readiness.

### 2. 🎯 Computer Vision (YOLO11n)
- **Object Detection**: Real-time identification of objects in images.
- **Inference & Fine-tuning**: Support for object detection, segmentation, and pose estimation using the latest YOLO models.
- **Real-time Processing**: Fast inference with confidence scores and annotated results.

### 3. 💬 AI Chatbot
- **Intelligent Conversations**: Powered by **ChatGroq (Llama 3.1)** for fast and context-aware responses.
- **History Management**: Track and export your AI interactions for future reference.

### 4. 😊 Sentiment Analysis
- **Emotion Detection**: Analyze text to identify positive, negative, or neutral tones using Hugging Face Transformers.
- **AI Feedback**: Get context-aware responses based on the analyzed sentiment.

### 5. 📝 Emotion-Based Poetry (Enhanced)
- **Real-time Face Detection**: Uses the camera to detect facial expressions (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral).
- **Bilingual Poem Generation**: AI generates beautiful 5-stanza poems in both **English** and **Urdu** simultaneously.
- **Proper Urdu Typography**: Integrated **Noto Nastaliq Urdu** font for an authentic reading experience.
- **Text-to-Speech (TTS)**: Integrated play buttons to listen to your poems in both English and Urdu voices.
- **Formatting**: Beautiful side-by-side layout with emotion-coded styling and explicit line-break management for Urdu script.

---

## 🛠️ Technology Stack

| Category | Libraries / Models |
| :--- | :--- |
| **Frontend/UI** | `Streamlit`, HTML5, CSS3 (Vanilla) |
| **Data Science** | `Pandas`, `NumPy`, `Scikit-learn`, `SciPy` |
| **Visualization** | `Plotly`, `Seaborn`, `Matplotlib` |
| **Computer Vision** | `OpenCV`, `Ultralytics (YOLOv8/v11)`, `Pillow`, `FER` |
| **AI & LLM** | `LangChain`, `ChatGroq (Llama 3.1)`, `Transformers`, `Torch` |
| **System/Audio** | `python-dotenv`, `Requests`, `gTTS`, `pydub` |

---

## 🚀 Steps Performed (Project Milestones)

1.  **Architecture Design**: Developed a modular folder structure to separate core logic, UI, and models.
2.  **Preprocessing Engine**: Built a full suite of transformer tools for automated data cleaning.
3.  **Vision Pipeline**: Successfully integrated YOLO models for diverse computer vision tasks.
4.  **Bilingual Art Module**:
    - Integrated **FER** for accurate facial expression recognition.
    - Optimized **ChatGroq** prompts to generate structured Urdu and English poetry.
    - Implemented a custom CSS engine to support **Right-to-Left (RTL)** Urdu text with Nastaliq fonts.
    - Added **Text-to-Speech** via `gTTS` with base64 audio streaming for instant playback.
5.  **Troubleshooting & Support**:
    - Resolved `ImportError` for `fer` package by mapping internal package structures.
    - Fixed Streamlit version compatibility issues (`use_container_width` vs `use_column_width`).
    - Standardized UI components across all 5 modules for a premium feel.

---

## ✅ Final Status
**Alhamdulillah! The AI Multi-Tool Platform is 100% functional, optimized, and tested.**

---

## 🔧 Installation

1. **Clone the repository:**
```bash
git clone <your-repo-link>
cd ai-multi-tool-platform
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up Environment Variables:**
Create a `.env` file or update `config.py` with your `GROQ_API_KEY`.

4. **Run the application:**
```bash
streamlit run app.py
```

---
*Made with ❤️ for AI Innovation*
=======
# 🚀 All-in-One AI Automation Platform

A powerful AI platform that combines multiple Artificial Intelligence domains into one unified application.

## 🌟 Features

### 📊 Data Analytics
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Interactive Dashboards
- Statistical Analysis
- Data Visualization
- Machine Learning
- Feature Engineering

---

### 🤖 Machine Learning

- Classification
- Regression
- Clustering
- Model Evaluation
- Hyperparameter Tuning
- Model Comparison

---

### 👁️ Computer Vision

- Image Classification
- Object Detection
- Face Detection
- Face Recognition
- Pose Estimation
- OCR
- Image Segmentation

---

### 💬 NLP

- Sentiment Analysis
- Text Classification
- Text Summarization
- Named Entity Recognition
- Keyword Extraction
- Translation

---

### 🤖 AI Chatbots

- RAG Chatbot
- PDF Chat
- Website Chat
- Document QA
- Memory Chatbot
- Multi-Agent Assistant

---

### 🎨 Generative AI

- Poetry Generator
- Story Generator
- Prompt Engineering
- Content Generation

---

### 🎥 Face AI

- Face Detection
- Face Verification
- Face Similarity
- Face Embeddings

---

### 📈 Time Series

- Forecasting
- Trend Analysis
- Anomaly Detection

---

### 🤖 AI Agents

- Supervisor Agent
- Planner Agent
- Data Analyst Agent
- Vision Agent
- NLP Agent
- Chatbot Agent
- Reviewer Agent

---

## 🛠️ Tech Stack

### Backend

- Python
- FastAPI
- Django

### AI

- Scikit-learn
- TensorFlow
- PyTorch
- Hugging Face
- LangChain
- LangGraph
- OpenAI
- FAISS
- ChromaDB

### Computer Vision

- OpenCV
- Ultralytics YOLO

### Frontend

- Streamlit
- React

### Database

- PostgreSQL
- SQLite

### Deployment

- Docker
- GitHub Actions

---

## Project Structure

```text
project/
│
├── backend/
├── frontend/
├── agents/
├── chatbot/
├── computer_vision/
├── data_analytics/
├── nlp/
├── generative_ai/
├── datasets/
├── models/
├── docs/
├── tests/
├── requirements.txt
└── README.md
```

---

## Future Features

- Voice Assistant
- AI Meeting Assistant
- Autonomous AI Agents
- Real-Time Video Analytics
- Multi-language Support
- Cloud Deployment
- Enterprise Authentication

---

## Author

Muhammad Usman

AI Engineer | AI & Data Science | Machine Learning | Computer Vision | NLP | Generative AI | Agentic AI
>>>>>>> 783fbab1e91492e71025da05eab8cbe342c6148a
