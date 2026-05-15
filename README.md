# ⚖️ Bangladesh Legal Aid Chatbot
### বাংলাদেশ আইন সহায়তা চ্যাটবট

> A free, end-to-end **Retrieval-Augmented Generation (RAG)** chatbot for Bangladesh law — runs entirely locally with no paid services.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![Groq](https://img.shields.io/badge/LLM-Groq%20%2B%20LLaMA%203-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🏗️ Architecture

```
PDFs (data/)
    │
    ▼
[PyMuPDF]  →  raw text per page
    │
    ▼
[RecursiveCharacterTextSplitter]  →  800-char chunks (150 overlap)
    │
    ▼
[HuggingFace sentence-transformers]  →  384-dim embeddings (local, free)
    │
    ▼
[ChromaDB]  →  local vector store (persisted to disk)
    │
    ▼  (at query time)
User Question  →  similarity search (Top-K chunks)
    │
    ▼
[Groq API + LLaMA 3 70B]  →  context-grounded answer
    │
    ▼
[Streamlit UI]  →  streamed response + source citations
```

---

## 🛠️ Tech Stack (100% Free)

| Component | Tool | Cost |
|-----------|------|------|
| PDF Parsing | PyMuPDF (`fitz`) | Free |
| Text Chunking | LangChain `RecursiveCharacterTextSplitter` | Free |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, local |
| Vector DB | ChromaDB (persistent, local) | Free |
| LLM | Groq API + LLaMA 3 70B | Free tier |
| Orchestration | LangChain LCEL | Free |
| Frontend | Streamlit | Free |

---

## 📁 Project Structure

```
bangladesh-legal-chatbot/
├── app.py                   # Streamlit chatbot UI
├── ingest.py                # One-time ingestion pipeline
├── rag_chain.py             # RAG chain (retrieval + generation)
├── config.py                # Centralised config from .env
├── test_setup.py            # Pre-flight setup checker
├── requirements.txt
├── .env.example             # Copy to .env and fill in
│
├── utils/
│   ├── pdf_loader.py        # PyMuPDF PDF parser
│   ├── chunker.py           # Text splitter
│   └── embedder.py          # HuggingFace embeddings + ChromaDB I/O
│
├── data/                    # ← Place your Bangladesh law PDFs here
│   └── README.txt
│
└── vectorstore/
    └── chroma_db/           # Auto-created by ingest.py
```

---

## 🚀 Quick Start

### Step 1 — Clone & Install

```bash
git clone https://github.com/your-username/bangladesh-legal-chatbot.git
cd bangladesh-legal-chatbot

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Step 2 — Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=gsk_your_key_here      # https://console.groq.com (free)
GROQ_MODEL=llama3-70b-8192
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DIR=./vectorstore/chroma_db
PDF_DIR=./data
CHUNK_SIZE=800
CHUNK_OVERLAP=150
TOP_K=5
```

### Step 3 — Add Bangladesh Law PDFs

Place one or more PDF files in the `data/` folder.

**Free official sources:**
| Document | URL |
|----------|-----|
| Constitution of Bangladesh | http://bdlaws.minlaw.gov.bd/act-367.html |
| Bangladesh Penal Code 1860 | http://bdlaws.minlaw.gov.bd/act-11.html |
| Code of Criminal Procedure 1898 | http://bdlaws.minlaw.gov.bd/act-75.html |
| Muslim Family Laws Ordinance 1961 | http://bdlaws.minlaw.gov.bd/act-319.html |
| Evidence Act 1872 | http://bdlaws.minlaw.gov.bd/act-24.html |
| Contract Act 1872 | http://bdlaws.minlaw.gov.bd/act-26.html |

### Step 4 — Verify Setup

```bash
python test_setup.py
```

### Step 5 — Build the Vector Store (run once)

```bash
python ingest.py
```

Output:
```
[1/3] Loading PDFs...     ✅ Loaded 312 pages from 3 PDF(s).
[2/3] Chunking...         ✅ Created 1,847 chunks
[3/3] Embedding & storing ✅ Vector store saved to ./vectorstore/chroma_db
✅ Ingestion complete in 94.3s — 1,847 chunks indexed.
```

> ⏱️ **First run downloads the embedding model (~90 MB)** — cached for all future runs.

### Step 6 — Launch the Chatbot

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 💬 Example Questions

- *"What are the fundamental rights guaranteed under the Bangladesh Constitution?"*
- *"What is the punishment for murder under the Bangladesh Penal Code?"*
- *"What are the grounds for divorce under Muslim family law?"*
- *"Explain the right to bail under the Code of Criminal Procedure."*
- *"What are the rules for land acquisition by the government?"*
- *"What constitutes wrongful confinement under Bangladesh law?"*

---

## ⚙️ Configuration Tuning

| Parameter | Default | Effect |
|-----------|---------|--------|
| `CHUNK_SIZE` | 800 | Larger = more context per chunk, slower embed |
| `CHUNK_OVERLAP` | 150 | Higher = fewer missed cross-chunk facts |
| `TOP_K` | 5 | More retrieved chunks = broader context |
| `GROQ_MODEL` | `llama3-70b-8192` | 70B = smarter but slower; 8B = faster |

---

## 🔄 Adding More PDFs Later

Just drop new PDFs into `data/` and re-run:
```bash
python ingest.py
```
This rebuilds the entire vector store with all documents.

---

## ⚠️ Disclaimer

> This chatbot is for **educational and informational purposes only**.
> It does not constitute legal advice. Always consult a qualified advocate
> for personal legal matters.

---

## 📄 License

MIT License — free to use, modify, and distribute.
