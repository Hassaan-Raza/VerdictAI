# VerdictAI
### Legal Document Intelligence

> Upload any legal document. Ask questions in plain English. Get cited, clause-referenced answers instantly.

![VerdictAI](https://img.shields.io/badge/VerdictAI-Legal%20Intelligence-C9A84C?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Cloud-black?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange?style=for-the-badge)

**Live Demo:** https://verdictai-9zjwujsqyk2gsccfdpf9an.streamlit.app/

---

## What is VerdictAI?

Most people sign contracts without fully understanding them. Not because they are careless, because legal language is deliberately complex. A single clause buried in section 5.1 can cap your entire liability recovery at $5,000 on a $148,500 contract.

VerdictAI is a RAG powered legal document intelligence system that reads any legal document and explains it in plain English. It identifies risky clauses, scores overall risk, maps obligations and deadlines, and answers any question you have about the document, all grounded in the actual text with clause references.

No legalese. No lawyer fees for basic comprehension. Any document, any jurisdiction.

---

## Features

**Chat**
Ask anything about your document in plain English and get cited, clause-referenced answers. Quick action chips for the most common questions: Summarize, My Obligations, My Rights, Red Flags, Deadlines, and Plain English.

**Analysis**
Seven one-click analysis modes:
- Plain Summary: document type, parties, purpose, and key terms
- Red Flag Detection: risky and one-sided clauses with HIGH / MEDIUM / LOW severity ratings
- Obligations: exactly what each party is required to do
- Rights & Entitlements: what each party is protected by
- Deadlines & Dates: every time-sensitive provision mapped out
- Party Analysis: detailed breakdown of each party and their position
- Risk Score: overall 1 to 10 risk rating with category breakdown across Financial, Legal, Operational, and Reputational risk

**Document View**
Word count, character count, chunk count, and full raw text for inspection.

---

## How It Works

```
User uploads PDF or TXT
        ↓
PyMuPDF extracts full text
        ↓
Text split into overlapping semantic chunks (800 words, 150 word overlap)
        ↓
HuggingFace all-MiniLM-L6-v2 generates embeddings
        ↓
Chunks indexed into ChromaDB vector store
        ↓
User asks a question
        ↓
Query embedded → vector similarity search → top 5 relevant chunks retrieved
        ↓
Retrieved chunks + question sent to Gemma 4 31B via Ollama Cloud
        ↓
Legal domain system prompt ensures clause citations and plain English output
        ↓
Answer returned with specific clause references
```

This is Retrieval Augmented Generation applied to law. Every answer is grounded in your actual document, not generated from memory.

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Gemma 4 31B via Ollama Cloud |
| Vector Store | ChromaDB |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| PDF Extraction | PyMuPDF (fitz) |
| Transcription | faster-whisper |
| Deployment | Streamlit Cloud |

---

## Getting Started

**Prerequisites**
- Python 3.11+
- An Ollama Cloud API key (free tier available at ollama.com)

**Installation**

```bash
git clone https://github.com/Hassaan-Raza/VerdictAI
cd VerdictAI
pip install -r requirements.txt
```

**Environment Setup**

Create a `.env` file in the root directory:

```
OLLAMA_API_KEY=your_ollama_api_key_here
OLLAMA_MODEL=gemma4:31b-cloud
```

Or if running on Streamlit Cloud, add these to your app secrets:

```toml
OLLAMA_API_KEY = "your_ollama_api_key_here"
OLLAMA_MODEL = "gemma4:31b-cloud"
```

**Run Locally**

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Project Structure

```
VerdictAI/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env                 # Local environment variables (not committed)
└── README.md            # This file
```

---

## Supported Documents

VerdictAI works with any legal document in PDF or TXT format:

- Contracts and service agreements
- NDAs and confidentiality agreements
- Employment agreements
- Leases and tenancy agreements
- Terms of service
- Court filings
- Legislation and regulations
- Any legal document from any jurisdiction

---

## Requirements

```
streamlit>=1.35.0
PyMuPDF>=1.24.0
sentence-transformers>=3.0.0
ollama>=0.4.0
chromadb==0.4.24
protobuf==3.20.3
python-dotenv>=1.0.0
faster-whisper>=1.0.0
```

---

## Disclaimer

VerdictAI provides legal information, not legal advice. It is designed to help people understand documents they are reading, not to replace a qualified attorney. Always consult a licensed legal professional before signing or acting on any legal document.

---

## Author

Built by **Hassaan Raza**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Hassaan%20Raza-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hassaan-raza-445)
[![GitHub](https://img.shields.io/badge/GitHub-Hassaan--Raza-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Hassaan-Raza)
