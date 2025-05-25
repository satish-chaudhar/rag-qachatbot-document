# README.md
## 🧠 RAG-Powered Document Q&A Chatbot

This is a full-stack Retrieval-Augmented Generation (RAG) app built with Streamlit, LangChain, and FAISS.

### 🚀 Features
- Multi-file upload (.txt, .pdf, .docx)
- Secure OpenAI key management (.env)
- FAISS vector database
- Semantic retrieval + OpenAI-powered answers
- Dockerized deployment-ready

### 📆 Run Locally
```bash
git clone https://github.com/yourname/rag_project.git
cd rag_project
cp .env.example .env  # Add your OpenAI Key
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app/main.py
```

### 🏠 Run via Docker
```bash
docker build -t rag-chatbot .
docker run -p 8501:8501 rag-chatbot
```

### ✈️ Production Checklist
- [x] Secure LLM API key
- [x] Clean document parsing
- [x] Token-efficient chunking
- [x] Accurate retrieval (top-k tuning)
- [x] Display source links
- [x] Responsive UI
