# 🤖 RAG Chatbot using LangChain, FastAPI & Streamlit

## 🚀 Overview

This project is an end-to-end **Retrieval-Augmented Generation (RAG) chatbot** that enables users to ask questions over multiple PDF documents. It combines LLM capabilities with vector-based retrieval to provide accurate and context-aware responses.

---

## 🧠 Key Features

* 🔍 Multi-document retrieval using FAISS
* 🧾 Source attribution (shows document reference)
* 💬 Conversational chatbot with memory
* ⚡ FastAPI backend for scalable API
* 🎨 Streamlit UI for interactive chat experience
* 🧠 Domain-based query routing (SQL / RAG / LLM)

---

## 🏗️ Architecture

User → Streamlit UI → FastAPI → RAG Pipeline → FAISS → LLM → Response

---

## 🛠️ Tech Stack

* LangChain
* FAISS (Vector DB)
* HuggingFace Embeddings
* Groq LLM
* FastAPI
* Streamlit
* Python

---

## 📂 Project Structure

```
rag-chatbot/
│
├── app.py              # FastAPI backend
├── ui.py               # Streamlit frontend
├── rag_pipeline.py     # Core RAG logic
├── requirements.txt
└── data/               # PDF documents
```

---

## ▶️ How to Run

### 1. Create virtual environment

```
conda create -p venv python==3.10 -y
conda activate venv/
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run backend

```
uvicorn app:app --reload
```

### 4. Run UI

```
streamlit run ui.py
```

---

## 💡 Example Use Cases

* Ask questions from technical PDFs
* Generate interview questions
* Understand concepts from documents

---

## ⚠️ Limitations

* Depends on quality of input documents
* Retrieval may miss relevant chunks if poorly indexed

---

## 🚀 Future Improvements

* Add citation highlighting
* Deploy on cloud (AWS / GCP)
* Add authentication

---
