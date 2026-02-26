# 🧠 Native RAG — Dark Psychology Research Assistant

A **Retrieval-Augmented Generation (RAG)** pipeline that lets you ask questions about a Dark Psychology research paper. Built with ChromaDB for vector storage and Groq's LLaMA 3.3 70B for intelligent responses.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-green)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-orange)

---

## 🏗️ Architecture

```
PDF Document
     │
     ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Extraction │────▶│   Chunking  │────▶│  Ingestion  │────▶│ Generation  │
│  (PyPDF)    │     │ (Recursive) │     │ (ChromaDB)  │     │ (Groq LLM)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                              │                    ▲
                                              │   Similarity       │
                                              └──── Search ────────┘
```

**Pipeline Steps:**
1. **Extraction** — Loads the PDF using `PyPDFLoader`
2. **Chunking** — Splits text into 1600-char chunks with 200-char overlap using `RecursiveCharacterTextSplitter`
3. **Ingestion** — Stores chunks as embeddings in a ChromaDB vector collection
4. **Retrieval** — Finds the top 3 most relevant chunks via similarity search
5. **Generation** — Sends context + query to LLaMA 3.3 70B via Groq, using the ROSE response framework

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com/)

### Installation

```bash
# Clone the repo
git clone https://github.com/muhaddasgujjar/Native-Rag.git
cd Native-Rag

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
pip install streamlit python-dotenv
```

### Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Run the App

```bash
# Streamlit UI (recommended)
streamlit run vector-db/app.py

# CLI mode
python vector-db/main.py
```

---

## 📁 Project Structure

```
Native-Rag/
├── .env                  # API key (not tracked by git)
├── .gitignore
├── requirements.txt
├── dark pyscology.pdf    # Source document
├── README.md
└── vector-db/
    ├── app.py            # Streamlit frontend
    ├── main.py           # CLI entry point
    ├── extraction.py     # PDF loading & text chunking
    ├── ingestion.py      # ChromaDB vector store operations
    ├── generation.py     # Groq LLM response generation
    ├── Rag.py            # Standalone single-file RAG script
    └── chroma.py         # ChromaDB exploration utilities
```

---

## 🖥️ Features

| Feature | Description |
|---|---|
| 💬 **Chat Interface** | Ask questions in a conversational UI powered by Streamlit |
| 📎 **Context Viewer** | Expand retrieved document chunks to see what the AI used |
| 🗄️ **Vector Search** | ChromaDB similarity search finds the most relevant passages |
| 🤖 **ROSE Framework** | Structured responses with Role, Objective, Specifics, and Examples |
| ⚡ **Groq Speed** | Ultra-fast inference via Groq's LPU hardware |
| 🔄 **Cached Pipeline** | PDF is processed once and cached across Streamlit reruns |

---

## 🛠️ Tech Stack

- **LLM**: LLaMA 3.3 70B via [Groq](https://groq.com/)
- **Vector DB**: [ChromaDB](https://www.trychroma.com/)
- **PDF Processing**: [LangChain Community](https://python.langchain.com/) + PyPDF
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Text Splitting**: LangChain `RecursiveCharacterTextSplitter`

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">Made with ❤️ by <a href="https://github.com/muhaddasgujjar">muhaddasgujjar</a></p>
