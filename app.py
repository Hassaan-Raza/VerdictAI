import streamlit as st
import os
import json
import tempfile
import time
import re
from pathlib import Path

import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import ollama
import faster_whisper

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="VerdictAI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --ink:      #0A0A0F;
  --paper:    #F5F0E8;
  --cream:    #EDE8DC;
  --gold:     #C9A84C;
  --gold2:    #E8C96A;
  --red:      #8B1A1A;
  --muted:    #6B6560;
  --border:   #D4CCB8;
  --mono:     'DM Mono', monospace;
  --serif:    'Playfair Display', serif;
  --sans:     'DM Sans', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--paper) !important;
  color: var(--ink) !important;
  font-family: var(--sans) !important;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse 60% 40% at 100% 0%, #C9A84C12 0%, transparent 60%),
    radial-gradient(ellipse 40% 30% at 0% 100%, #8B1A1A08 0%, transparent 50%),
    var(--paper) !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--ink) !important;
  border-right: 1px solid #1E1E28 !important;
}
[data-testid="stSidebar"] label {
  color: #6B6560 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
  color: #C8C0B0 !important;
  font-family: var(--mono) !important;
}

/* ── Main content text ── */
[data-testid="stMain"] p,
[data-testid="stMain"] span,
[data-testid="stMain"] li,
[data-testid="stMain"] div,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span {
  color: var(--ink) !important;
  font-family: var(--sans) !important;
}

.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  font-family: var(--mono) !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.08em !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--ink) !important;
}
.stTabs [aria-selected="true"] {
  color: var(--ink) !important;
  border-bottom: 2px solid var(--gold) !important;
}
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span {
  color: inherit !important;
}

/* ── Buttons ── */
.stButton > button {
  background: var(--ink) !important;
  color: var(--gold) !important;
  border: 1px solid var(--gold) !important;
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  border-radius: 2px !important;
  padding: 0.5rem 1.2rem !important;
  transition: all 0.2s !important;
}
.stButton > button p,
.stButton > button span {
  color: var(--gold) !important;
  font-family: var(--mono) !important;
}
.stButton > button:hover {
  background: var(--gold) !important;
  color: var(--ink) !important;
}
.stButton > button:hover p,
.stButton > button:hover span {
  color: var(--ink) !important;
}

/* Primary button */
[data-testid="baseButton-primary"] > button {
  background: var(--gold) !important;
  color: var(--ink) !important;
  font-weight: 600 !important;
  border-color: var(--gold) !important;
}
[data-testid="baseButton-primary"] > button p,
[data-testid="baseButton-primary"] > button span {
  color: var(--ink) !important;
}

/* ── Text area ── */
.stTextArea textarea {
  background: var(--cream) !important;
  border: 1px solid var(--border) !important;
  border-radius: 2px !important;
  font-family: var(--sans) !important;
  font-size: 0.9rem !important;
  color: var(--ink) !important;
}
.stTextArea textarea:focus {
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 2px #C9A84C20 !important;
}
.stTextArea textarea::placeholder { color: #A09890 !important; }

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
  background: var(--cream) !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: 2px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--gold) !important;
}
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span {
  color: var(--muted) !important;
}

/* ── Labels ── */
label {
  font-family: var(--mono) !important;
  font-size: 0.68rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  color: var(--muted) !important;
}

/* ── Selectbox ── */
.stSelectbox div[data-baseweb="select"] > div {
  background: var(--cream) !important;
  border-color: var(--border) !important;
  font-family: var(--sans) !important;
  border-radius: 2px !important;
  color: var(--ink) !important;
}

/* ── Success / info boxes ── */
[data-testid="stAlert"] p { color: inherit !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--paper); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Chat messages ── */
.verdict-msg-user {
  background: var(--ink);
  color: var(--paper) !important;
  border-radius: 2px;
  padding: 1rem 1.2rem;
  margin-bottom: 0.8rem;
  font-family: var(--sans);
  font-size: 0.9rem;
  line-height: 1.6;
  border-left: 3px solid var(--gold);
}
.verdict-msg-ai {
  background: var(--cream);
  color: var(--ink) !important;
  border-radius: 2px;
  padding: 1rem 1.2rem;
  margin-bottom: 0.8rem;
  font-family: var(--sans);
  font-size: 0.9rem;
  line-height: 1.7;
  border-left: 3px solid var(--red);
  white-space: pre-wrap;
}
.verdict-disclaimer {
  background: #8B1A1A10;
  border: 1px solid #8B1A1A30;
  border-radius: 2px;
  padding: 0.7rem 1rem;
  font-family: var(--mono);
  font-size: 0.68rem;
  color: var(--red) !important;
  line-height: 1.6;
  margin-top: 1rem;
}
.verdict-result {
  background: var(--cream);
  border: 1px solid var(--border);
  border-radius: 2px;
  padding: 1.5rem 2rem;
  font-family: var(--sans);
  font-size: 0.9rem;
  line-height: 1.8;
  color: var(--ink) !important;
  white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 3rem 1rem; border-bottom: 1px solid #D4CCB8;">
  <div style="display:flex; align-items:baseline; gap:1rem; margin-bottom:0.3rem;">
    <span style="font-family:'Playfair Display',serif; font-size:2.6rem; font-weight:700;
                 color:#0A0A0F; letter-spacing:-0.02em; line-height:1;">
      Verdict<span style="color:#C9A84C; font-style:italic;">AI</span>
    </span>
    <span style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#6B6560;
                 letter-spacing:0.15em; text-transform:uppercase;
                 border:1px solid #D4CCB8; padding:2px 8px; border-radius:2px;">
      Legal Intelligence
    </span>
  </div>
  <p style="font-family:'DM Mono',monospace; font-size:0.72rem; color:#6B6560;
            margin:0; letter-spacing:0.04em;">
    Upload any legal document · Ask questions in plain English · Get cited answers · Globally applicable
  </p>
</div>
""", unsafe_allow_html=True)

# ── Config — paste your Ollama API key below ──────────────────
OLLAMA_API_KEY  = "30e05816dd89460f9281057d786d25fe.E6H3Ni8qsL8MX2EZuuP_Gn9M"
CHROMA_HOST     = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT     = int(os.getenv("CHROMA_PORT", "8000"))
OLLAMA_MODEL = "gemma4:31b-cloud"
EMBED_MODEL     = "all-MiniLM-L6-v2"

# ── Clients ───────────────────────────────────────────────────
@st.cache_resource
def get_ollama_client():
    return ollama.Client(
        host="https://api.ollama.com",
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
    )

@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource
def get_chroma():
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat()
        return client
    except Exception:
        return chromadb.Client()

@st.cache_resource
def get_whisper():
    return faster_whisper.WhisperModel("base", device="cpu", compute_type="int8")

# ── Document processing ───────────────────────────────────────
def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        doc  = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    return file_bytes.decode("utf-8", errors="ignore")

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list:
    words  = text.split()
    chunks = []
    i      = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 50]

def index_document(text: str, doc_id: str):
    embedder   = get_embedder()
    chroma     = get_chroma()
    collection = chroma.get_or_create_collection(f"verdict_{doc_id}")
    chunks     = chunk_text(text)
    embeddings = embedder.encode(chunks).tolist()
    ids        = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return collection, chunks

def retrieve(query: str, collection, top_k: int = 5) -> list:
    embedder = get_embedder()
    qemb     = embedder.encode([query]).tolist()
    results  = collection.query(query_embeddings=qemb, n_results=min(top_k, collection.count()))
    return results["documents"][0] if results["documents"] else []

# ── LLM ───────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are VerdictAI, a highly knowledgeable legal assistant. You help people understand legal documents in plain English.

Rules:
- Always cite specific clauses or sections when answering
- Use plain, accessible language — not legalese
- Flag any risky or unusual clauses clearly
- Never give definitive legal advice — always recommend consulting a qualified attorney for important decisions
- Be globally applicable — do not assume a specific jurisdiction unless the document specifies one
- Be thorough but concise
- Structure your answers with clear headings when appropriate"""

def ask_llm(question: str, context_chunks: list, chat_history: list) -> str:
    client  = get_ollama_client()
    context = "\n\n---\n\n".join(context_chunks)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in chat_history[-6:]:
        messages.append(msg)
    messages.append({
        "role": "user",
        "content": f"Relevant document excerpts:\n{context}\n\nQuestion: {question}\n\nAnswer in plain English with specific references to the document content above."
    })
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
    )
    result = response.message.content
    result = re.sub(r'^#{1,6}\s+', '', result, flags=re.MULTILINE)
    return result

def analyze_document(text: str, analysis_type: str) -> str:
    client  = get_ollama_client()
    prompts = {
        "summary":      f"Provide a clear plain-English summary of this legal document. Include: document type, parties involved, main purpose, key terms, and important dates/deadlines.\n\nDocument:\n{text[:6000]}",
        "red_flags":    f"Analyze this legal document and identify ALL potentially risky, unusual, or unfavorable clauses. For each red flag: name the clause, explain the risk in plain English, and rate severity as LOW/MEDIUM/HIGH.\n\nDocument:\n{text[:6000]}",
        "obligations":  f"Extract ALL obligations and responsibilities from this legal document. List what each party is REQUIRED to do, with clause references.\n\nDocument:\n{text[:6000]}",
        "rights":       f"Extract ALL rights and entitlements from this legal document. List what each party is ENTITLED to, with clause references.\n\nDocument:\n{text[:6000]}",
        "deadlines":    f"Extract ALL dates, deadlines, timeframes, and time-sensitive provisions from this legal document. Format as a clear timeline.\n\nDocument:\n{text[:6000]}",
        "parties":      f"Identify all parties in this legal document. For each party: their name/role, their obligations, their rights, and any special conditions that apply to them.\n\nDocument:\n{text[:6000]}",
        "risk_score":   f"Analyze this legal document and provide:\n1. An overall RISK SCORE from 1-10 (10 being highest risk)\n2. Risk breakdown by category (Financial, Legal, Operational, Reputational)\n3. Top 3 concerns\n4. Overall recommendation\n\nDocument:\n{text[:6000]}",
    }
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompts.get(analysis_type, prompts["summary"])}
        ],
    )
    result = response.message.content
    result = re.sub(r'^#{1,6}\s+', '', result, flags=re.MULTILINE)
    return result

def transcribe_audio(audio_bytes: bytes) -> str:
    whisper = get_whisper()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    segments, _ = whisper.transcribe(tmp_path, beam_size=5)
    text = " ".join(seg.text for seg in segments).strip()
    os.unlink(tmp_path)
    return text

# ── Session state ─────────────────────────────────────────────
if "doc_text"       not in st.session_state: st.session_state.doc_text       = None
if "doc_id"         not in st.session_state: st.session_state.doc_id         = None
if "collection"     not in st.session_state: st.session_state.collection     = None
if "chat_history"   not in st.session_state: st.session_state.chat_history   = []
if "doc_text_2"     not in st.session_state: st.session_state.doc_text_2     = None
if "analysis_cache" not in st.session_state: st.session_state.analysis_cache = {}

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.5rem;">
      <div style="font-family:'Playfair Display',serif; font-size:1.1rem; font-weight:600;
                  color:#C9A84C; margin-bottom:0.25rem;">VerdictAI</div>
      <div style="font-family:'DM Mono',monospace; font-size:0.62rem; color:#6B6560;
                  text-transform:uppercase; letter-spacing:0.1em;">Legal Intelligence Suite</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div style="font-family:'DM Mono',monospace; font-size:0.62rem; color:#6B6560;
                text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.6rem;">
                Upload Document</div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload legal document", type=["pdf", "txt"],
                                label_visibility="collapsed")

    if uploaded:
        if st.button("⚖ Process Document", use_container_width=True, type="primary"):
            with st.spinner("Extracting and indexing..."):
                raw    = uploaded.read()
                text   = extract_text(raw, uploaded.name)
                doc_id = uploaded.name.replace(" ", "_").replace(".", "_")
                col, _ = index_document(text, doc_id)
                st.session_state.doc_text       = text
                st.session_state.doc_id         = doc_id
                st.session_state.collection     = col
                st.session_state.chat_history   = []
                st.session_state.analysis_cache = {}
            st.success(f"Ready — {len(text):,} characters indexed")

    if st.session_state.doc_text:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'DM Mono',monospace; font-size:0.62rem; color:#6B6560;
                    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.6rem;">
                    Compare Documents</div>""", unsafe_allow_html=True)
        uploaded_2 = st.file_uploader("Second document", type=["pdf", "txt"],
                                       label_visibility="collapsed", key="doc2")
        if uploaded_2:
            raw2 = uploaded_2.read()
            st.session_state.doc_text_2 = extract_text(raw2, uploaded_2.name)
            st.success("Second document loaded")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:0.62rem; line-height:2;">
      <span style="color:#C9A84C;">Model</span><br>
      <span style="color:#4A4A4A;">Gemini 3 Flash via Ollama</span><br><br>
      <span style="color:#C9A84C;">Embeddings</span><br>
      <span style="color:#4A4A4A;">all-MiniLM-L6-v2</span><br><br>
      <span style="color:#C9A84C;">Vector Store</span><br>
      <span style="color:#4A4A4A;">ChromaDB</span><br><br>
      <span style="color:#C9A84C;">Voice</span><br>
      <span style="color:#4A4A4A;">Whisper Base</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:0.6rem; color:#3A3A3A; line-height:1.7;">
      VerdictAI provides legal information, not legal advice.
      Always consult a qualified attorney for important legal decisions.
    </div>
    """, unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────
pad = "padding: 2rem 3rem;"

if not st.session_state.doc_text:
    st.markdown(f"""
    <div style="{pad}">
      <div style="max-width:600px; margin:4rem auto; text-align:center;">
        <div style="font-family:'Playfair Display',serif; font-size:3.5rem; color:#D4CCB8;
                    margin-bottom:1rem; font-style:italic;">⚖</div>
        <div style="font-family:'Playfair Display',serif; font-size:1.4rem; color:#6B6560;
                    margin-bottom:0.8rem;">Upload a legal document to begin</div>
        <div style="font-family:'DM Mono',monospace; font-size:0.75rem; color:#A09890; line-height:1.8;">
          Contracts · Legislation · Court filings · NDAs · Terms of service<br>
          Leases · Employment agreements · Any legal document, any jurisdiction
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🔍 Analysis", "📊 Compare", "📄 Document"])

    # ── CHAT ──────────────────────────────────────────────────
    with tab1:
        st.markdown(f'<div style="{pad} padding-bottom:0;">', unsafe_allow_html=True)

        quick_qs = {
            "Summarize":      "What is this document about and what are its main terms?",
            "My obligations": "What am I required to do under this agreement?",
            "My rights":      "What rights and entitlements do I have under this document?",
            "Red flags":      "Are there any risky or unfavorable clauses I should know about?",
            "Deadlines":      "What are all the important dates and deadlines in this document?",
            "Plain English":  "Explain this entire document in simple plain English as if I have no legal knowledge.",
        }

        qcols = st.columns(len(quick_qs))
        for i, (label, question) in enumerate(quick_qs.items()):
            if qcols[i].button(label, key=f"q_{i}", use_container_width=True):
                st.session_state["pending_question"] = question
                st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="verdict-msg-user">🧑 {msg["content"]}</div>',
                            unsafe_allow_html=True)
            else:
                with st.container():
                    st.markdown(f"⚖ {msg['content']}")

        if "pending_question" in st.session_state:
            st.session_state["current_input"] = st.session_state.pop("pending_question")

        col_input, col_voice = st.columns([5, 1])
        with col_input:
            user_input = st.text_area(
                "Ask",
                value=st.session_state.get("current_input", ""),
                height=90,
                label_visibility="collapsed",
                placeholder="Ask anything about this document in plain English..."
            )
            st.session_state["current_input"] = user_input
        with col_voice:
            pass

        ask_btn = st.button("⚖ Ask VerdictAI", type="primary", use_container_width=True)

        if ask_btn and user_input.strip():
            with st.spinner("Retrieving relevant clauses and generating answer..."):
                chunks   = retrieve(user_input, st.session_state.collection)
                response = ask_llm(user_input, chunks, st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "user",     "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.session_state["current_input"] = ""
            st.rerun()

        if st.session_state.chat_history:
            if st.button("Clear conversation", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # ── ANALYSIS ──────────────────────────────────────────────
    with tab2:
        st.markdown(f'<div style="{pad}">', unsafe_allow_html=True)

        analyses = {
            "📋 Plain Summary":        "summary",
            "🚨 Red Flag Detection":   "red_flags",
            "📌 Obligations":          "obligations",
            "✅ Rights & Entitlements":"rights",
            "📅 Deadlines & Dates":    "deadlines",
            "👥 Party Analysis":       "parties",
            "📊 Risk Score":           "risk_score",
        }

        acols = st.columns(4)
        selected_analysis = None
        for i, (label, key) in enumerate(analyses.items()):
            if acols[i % 4].button(label, key=f"an_{key}", use_container_width=True):
                selected_analysis = key

        st.markdown("<hr>", unsafe_allow_html=True)

        if selected_analysis:
            if selected_analysis not in st.session_state.analysis_cache:
                with st.spinner("Analyzing document..."):
                    result = analyze_document(st.session_state.doc_text, selected_analysis)
                    st.session_state.analysis_cache[selected_analysis] = result

            result = st.session_state.analysis_cache[selected_analysis]
            with st.container():
                st.markdown(result)
            st.markdown("""<div class="verdict-disclaimer">
              ⚠ This analysis is generated by AI and is for informational purposes only.
              It does not constitute legal advice. Always consult a qualified attorney before
              making decisions based on this analysis.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style="text-align:center; padding:3rem; color:#A09890;
                        font-family:'DM Mono',monospace; font-size:0.78rem;">
                        Select an analysis type above to begin</div>""",
                        unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── COMPARE ───────────────────────────────────────────────
    with tab3:
        st.markdown(f'<div style="{pad}">', unsafe_allow_html=True)

        if not st.session_state.doc_text_2:
            st.markdown("""<div style="text-align:center; padding:3rem; color:#A09890;
                        font-family:'DM Mono',monospace; font-size:0.78rem;">
                        Upload a second document in the sidebar to compare</div>""",
                        unsafe_allow_html=True)
        else:
            if st.button("⚖ Compare Documents", type="primary", use_container_width=True):
                with st.spinner("Comparing documents..."):
                    client = get_ollama_client()
                    compare_prompt = f"""Compare these two legal documents thoroughly. Identify:
1. Key differences in obligations
2. Key differences in rights
3. Which document is more favorable and why
4. Unique clauses in each document
5. Risk comparison between the two

Document 1:\n{st.session_state.doc_text[:3000]}

Document 2:\n{st.session_state.doc_text_2[:3000]}"""
                    response = client.chat(
                        model=OLLAMA_MODEL,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": compare_prompt}
                        ],
                    )
                    st.session_state.analysis_cache["compare"] = response.message.content

            if "compare" in st.session_state.analysis_cache:
                st.markdown(st.session_state.analysis_cache["compare"])
                st.markdown("""<div class="verdict-disclaimer">
                  ⚠ This comparison is generated by AI for informational purposes only.
                  Always consult a qualified attorney before making decisions.
                </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── DOCUMENT ──────────────────────────────────────────────
    with tab4:
        st.markdown(f'<div style="{pad}">', unsafe_allow_html=True)

        word_count = len(st.session_state.doc_text.split())
        char_count = len(st.session_state.doc_text)
        chunks     = chunk_text(st.session_state.doc_text)

        st.markdown(f"""
        <div style="display:flex; gap:2rem; margin-bottom:1.5rem;">
          <div style="background:var(--cream);border:1px solid var(--border);border-radius:2px;padding:1rem 1.5rem;">
            <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#6B6560;text-transform:uppercase;letter-spacing:0.1em;">Words</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.8rem;color:#C9A84C;">{word_count:,}</div>
          </div>
          <div style="background:var(--cream);border:1px solid var(--border);border-radius:2px;padding:1rem 1.5rem;">
            <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#6B6560;text-transform:uppercase;letter-spacing:0.1em;">Characters</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.8rem;color:#C9A84C;">{char_count:,}</div>
          </div>
          <div style="background:var(--cream);border:1px solid var(--border);border-radius:2px;padding:1rem 1.5rem;">
            <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#6B6560;text-transform:uppercase;letter-spacing:0.1em;">Chunks indexed</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.8rem;color:#C9A84C;">{len(chunks)}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.text_area("Raw document text", value=st.session_state.doc_text,
                     height=400, label_visibility="collapsed")

        st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1rem 3rem; border-top:1px solid #D4CCB8; margin-top:2rem;
            display:flex; justify-content:space-between; align-items:center;">
  <span style="font-family:'DM Mono',monospace; font-size:0.62rem; color:#A09890;">
    VerdictAI · Legal Intelligence · Globally applicable
  </span>
  <span style="font-family:'DM Mono',monospace; font-size:0.62rem; color:#A09890;">
    Not legal advice · Consult a qualified attorney for important decisions
  </span>
</div>
""", unsafe_allow_html=True)
