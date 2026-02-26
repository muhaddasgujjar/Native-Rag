#  Native RAG — Dark Psychology Research Assistant

A simple **Retrieval-Augmented Generation** pipeline that answers questions about a Dark Psychology PDF using ChromaDB + Groq LLaMA 3.3.

---

## 1. Essential Setup Commands

Open your terminal in the project folder and run these:

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Install the core RAG stack
pip install chromadb groq langchain-community pypdf langchain-text-splitters python-dotenv

# Install Streamlit (for the web UI)
pip install streamlit
```

---

## 2. Environment Configuration

Create a `.env` file in the project root with your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get your free API key from [console.groq.com](https://console.groq.com/)

---

## 3. File Organization

Make sure your folder looks exactly like this before running:

```
Native-Rag/
├── .env                  # Your API key (not pushed to git)
├── requirements.txt
├── dark pyscology.pdf    # Your research paper
└── vector-db/
    ├── app.py            # Streamlit frontend (web UI)
    ├── main.py           # CLI entry point
    ├── extraction.py     # The PDF reader
    ├── ingestion.py      # The Database manager
    └── generation.py     # The LLM / ROSE logic
```

---

## 4. Execution Guide

**Option A — Streamlit UI (recommended):**

```bash
streamlit run vector-db/app.py
```

**Option B — CLI mode:**

```bash
python vector-db/main.py
```

---

## 5. Step-by-Step Testing Guide

When you run the app, follow this sequence to verify everything works:

**Step 1: Check Ingestion**
Watch the terminal. It should say:
```
Successfully ingested X document chunks.
```

**Step 2: Simple Query**
Ask: *"What is the main topic of this paper?"*
→ Verifies the retriever is pulling correct context.

**Step 3: Persona Check**
Ask: *"Explain Dark Triad traits."*
→ Verify the response follows the ROSE framework (Role, Objective, Specifics, Examples).

**Step 4: Hallucination Check**
Ask: *"What is the recipe for chocolate cake?"*
→ The bot should say it doesn't have the answer based on the context.

---

## 6. Git Commands (Push to GitHub)

Once testing is finished, use these commands to upload your work:

```bash
git init
git add .
git commit -m "Initial commit: Modular RAG with ROSE framework"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | LLaMA 3.3 70B via Groq |
| Vector DB | ChromaDB |
| PDF Loader | LangChain + PyPDF |
| Frontend | Streamlit |
| Framework | ROSE (Role, Objective, Specifics, Examples) |

---

<p align="center">Made with ❤️ by <a href="https://github.com/muhaddasgujjar">muhaddasgujjar</a></p>
