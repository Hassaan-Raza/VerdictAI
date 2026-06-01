import streamlit as st
import os
import tempfile
import time
import re

import fitz
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import faster_whisper

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="VerdictAI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Fonts ─────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
  --ink:    #0A0A0F;
  --paper:  #F5F0E8;
  --cream:  #EDE8DC;
  --gold:   #C9A84C;
  --red:    #8B1A1A;
  --muted:  #6B6560;
  --border: #D4CCB8;
  --mono:   'DM Mono', monospace;
  --serif:  'Playfair Display', serif;
  --sans:   'DM Sans', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--paper) !important;
  color: var(--ink) !important;
  font-family: 'DM Sans', sans-serif !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* hide default sidebar toggle */
[data-testid="collapsedControl"] { display: none !important; }

.block-container { padding: 0 !important; max-width: 100% !important; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--ink) !important;
  border-right: 1px solid #1E1E28 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #C8C0B0 !important; font-family: var(--mono) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--muted) !important; font-family: var(--mono) !important; font-size: 0.75rem !important; }
.stTabs [aria-selected="true"] { color: var(--ink) !important; border-bottom: 2px solid var(--gold) !important; }
.stTabs [data-baseweb="tab"] p, .stTabs [data-baseweb="tab"] span { color: inherit !important; }

/* Buttons */
.stButton > button {
  background: var(--ink) !important; color: var(--gold) !important;
  border: 1px solid var(--gold) !important; font-family: var(--mono) !important;
  font-size: 0.72rem !important; letter-spacing: 0.12em !important;
  text-transform: uppercase !important; border-radius: 2px !important;
  padding: 0.5rem 1.2rem !important; transition: all 0.2s !important;
}
.stButton > button p, .stButton > button span { color: var(--gold) !important; }
.stButton > button:hover { background: var(--gold) !important; color: var(--ink) !important; }
.stButton > button:hover p, .stButton > button:hover span { color: var(--ink) !important; }

[data-testid="baseButton-primary"] > button {
  background: var(--gold) !important; color: var(--ink) !important;
  font-weight: 600 !important; border-color: var(--gold) !important;
}
[data-testid="baseButton-primary"] > button p,
[data-testid="baseButton-primary"] > button span { color: var(--ink) !important; }

/* Text area */
.stTextArea textarea {
  background: var(--cream) !important; border: 1px solid var(--border) !important;
  border-radius: 2px !important; font-family: 'DM Sans', sans-serif !important;
  font-size: 0.9rem !important; color: var(--ink) !important;
}
.stTextArea textarea:focus { border-color: var(--gold) !important; }
.stTextArea textarea::placeholder { color: #A09890 !important; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
  background: var(--cream) !important; border: 1.5px dashed var(--border) !important;
  border-radius: 2px !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--gold) !important; }
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span { color: var(--muted) !important; }

label { font-family: var(--mono) !important; font-size: 0.68rem !important;
        text-transform: uppercase !important; letter-spacing: 0.1em !important;
        color: var(--muted) !important; }

hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--paper); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* Chat messages */
.verdict-msg-user {
  background: var(--ink); color: var(--paper) !important;
  border-radius: 2px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
  font-family: 'DM Sans', sans-serif; font-size: 0.9rem; line-height: 1.6;
  border-left: 3px solid var(--gold);
}
.verdict-disclaimer {
  background: #8B1A1A10; border: 1px solid #8B1A1A30; border-radius: 2px;
  padding: 0.7rem 1rem; font-family: var(--mono); font-size: 0.68rem;
  color: var(--red) !important; line-height: 1.6; margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────
OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY", os.getenv("OLLAMA_API_KEY", ""))
OLLAMA_MODEL   = st.secrets.get("OLLAMA_MODEL", "gemma4:31b-cloud")
CHROMA_HOST    = st.secrets.get("CHROMA_HOST", os.getenv("CHROMA_HOST", "localhost"))
CHROMA_PORT    = int(st.secrets.get("CHROMA_PORT", os.getenv("CHROMA_PORT", "8000")))
EMBED_MODEL    = "all-MiniLM-L6-v2"

# ── Clients ───────────────────────────────────────────────────
@st.cache_resource
def get_ollama_client():
    return ollama.Client(host="https://api.ollama.com",
                         headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"})

@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource
def get_chroma():
    try:
        c = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        c.heartbeat()
        return c
    except Exception:
        return chromadb.Client()

@st.cache_resource
def get_whisper():
    return faster_whisper.WhisperModel("base", device="cpu", compute_type="int8")

# ── Helpers ───────────────────────────────────────────────────
def extract_text(file_bytes, filename):
    if filename.lower().endswith(".pdf"):
        doc  = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n\n".join(p.get_text() for p in doc)
        doc.close()
        return text
    return file_bytes.decode("utf-8", errors="ignore")

def chunk_text(text, chunk_size=800, overlap=150):
    words, chunks, i = text.split(), [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 50]

def index_document(text, doc_id):
    embedder   = get_embedder()
    chroma     = get_chroma()
    collection = chroma.get_or_create_collection(f"verdict_{doc_id}")
    chunks     = chunk_text(text)
    embeddings = embedder.encode(chunks).tolist()
    ids        = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return collection

def retrieve(query, collection, top_k=5):
    embedder = get_embedder()
    qemb     = embedder.encode([query]).tolist()
    results  = collection.query(query_embeddings=qemb, n_results=min(top_k, collection.count()))
    return results["documents"][0] if results["documents"] else []

SYSTEM_PROMPT = """You are VerdictAI, a highly knowledgeable legal assistant.
- Always cite specific clauses when answering
- Use plain accessible language, not legalese
- Flag risky clauses clearly
- Never give definitive legal advice — recommend consulting an attorney
- Be globally applicable — don't assume jurisdiction unless document specifies
- Be thorough but concise"""

def ask_llm(question, context_chunks, chat_history):
    client   = get_ollama_client()
    context  = "\n\n---\n\n".join(context_chunks)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in chat_history[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": f"Document excerpts:\n{context}\n\nQuestion: {question}\n\nAnswer in plain English with document references."})
    response = client.chat(model=OLLAMA_MODEL, messages=messages)
    return re.sub(r'^#{1,6}\s+', '', response.message.content, flags=re.MULTILINE)

def analyze_document(text, analysis_type):
    client  = get_ollama_client()
    prompts = {
        "summary":     f"Provide a clear plain-English summary. Include: document type, parties, main purpose, key terms, important dates.\n\nDocument:\n{text[:6000]}",
        "red_flags":   f"Identify ALL risky, unusual, or unfavorable clauses. For each: name it, explain the risk in plain English, rate severity LOW/MEDIUM/HIGH.\n\nDocument:\n{text[:6000]}",
        "obligations": f"Extract ALL obligations. List what each party is REQUIRED to do, with clause references.\n\nDocument:\n{text[:6000]}",
        "rights":      f"Extract ALL rights and entitlements. List what each party is ENTITLED to, with clause references.\n\nDocument:\n{text[:6000]}",
        "deadlines":   f"Extract ALL dates, deadlines, and time-sensitive provisions. Format as a clear timeline.\n\nDocument:\n{text[:6000]}",
        "parties":     f"Identify all parties. For each: name/role, obligations, rights, special conditions.\n\nDocument:\n{text[:6000]}",
        "risk_score":  f"Provide:\n1. RISK SCORE 1-10\n2. Risk breakdown by category (Financial, Legal, Operational, Reputational)\n3. Top 3 concerns\n4. Overall recommendation\n\nDocument:\n{text[:6000]}",
    }
    response = client.chat(model=OLLAMA_MODEL, messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompts.get(analysis_type, prompts["summary"])}
    ])
    return re.sub(r'^#{1,6}\s+', '', response.message.content, flags=re.MULTILINE)

# ── Session state ─────────────────────────────────────────────
for k, v in [("doc_text", None), ("doc_id", None), ("collection", None),
              ("chat_history", []), ("doc_text_2", None), ("analysis_cache", {}),
              ("show_info", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1.8rem 3rem 1rem; border-bottom: 1px solid #D4CCB8;
            display:flex; align-items:center; justify-content:space-between;">
  <div>
    <div style="display:flex; align-items:baseline; gap:1rem; margin-bottom:0.2rem;">
      <span style="font-family:'Playfair Display',serif; font-size:2.4rem; font-weight:700;
                   color:#0A0A0F; letter-spacing:-0.02em; line-height:1;">
        Verdict<span style="color:#C9A84C; font-style:italic;">AI</span>
      </span>
      <span style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#6B6560;
                   letter-spacing:0.15em; text-transform:uppercase;
                   border:1px solid #D4CCB8; padding:2px 8px; border-radius:2px;">
        Legal Intelligence
      </span>
    </div>
    <p style="font-family:'DM Mono',monospace; font-size:0.72rem; color:#6B6560; margin:0;">
      Upload any legal document · Ask in plain English · Get cited answers · Any jurisdiction
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Main content ──────────────────────────────────────────────
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
pad = "padding: 1.5rem 3rem;"
if st.session_state.document_uploaded:
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
        tab1, tab2, tab4 = st.tabs(["💬 Chat", "🔍 Analysis", "📄 Document"])
        with tab1:
            st.markdown(f'<div style="{pad} padding-bottom:0;">', unsafe_allow_html=True)

            quick_qs = {
                "Summarize": "What is this document about and what are its main terms?",
                "My obligations": "What am I required to do under this agreement?",
                "My rights": "What rights and entitlements do I have under this document?",
                "Red flags": "Are there any risky or unfavorable clauses I should know about?",
                "Deadlines": "What are all the important dates and deadlines in this document?",
                "Plain English": "Explain this entire document in simple plain English.",
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

            user_input = st.text_area(
                "Ask",
                value=st.session_state.get("current_input", ""),
                height=90,
                label_visibility="collapsed",
                placeholder="Ask anything about this document in plain English..."
            )
            st.session_state["current_input"] = user_input

            ask_btn = st.button("⚖ Ask VerdictAI", type="primary", use_container_width=True)

            if ask_btn and user_input.strip():
                with st.spinner("Retrieving relevant clauses and generating answer..."):
                    chunks = retrieve(user_input, st.session_state.collection)
                    response = ask_llm(user_input, chunks, st.session_state.chat_history)
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.session_state["current_input"] = ""
                st.rerun()

            if st.session_state.chat_history:
                if st.button("Clear conversation", use_container_width=True):
                    st.session_state.chat_history = []
                    st.session_state.document_uploaded = False
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        # ── ANALYSIS ──────────────────────────────────────────────
        with tab2:
            st.markdown(f'<div style="{pad}">', unsafe_allow_html=True)

            analyses = {
                "📋 Plain Summary": "summary",
                "🚨 Red Flag Detection": "red_flags",
                "📌 Obligations": "obligations",
                "✅ Rights & Entitlements": "rights",
                "📅 Deadlines & Dates": "deadlines",
                "👥 Party Analysis": "parties",
                "📊 Risk Score": "risk_score",
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
                st.markdown(st.session_state.analysis_cache[selected_analysis])
                st.markdown("""<div class="verdict-disclaimer">
                      ⚠ This analysis is generated by AI for informational purposes only.
                      It does not constitute legal advice. Always consult a qualified attorney.
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div style="text-align:center; padding:3rem; color:#A09890;
                                font-family:'DM Mono',monospace; font-size:0.78rem;">
                                Select an analysis type above to begin</div>""",
                            unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # ── DOCUMENT ──────────────────────────────────────────────
        with tab4:
            st.markdown(f'<div style="{pad}">', unsafe_allow_html=True)

            word_count = len(st.session_state.doc_text.split())
            char_count = len(st.session_state.doc_text)
            chunks = chunk_text(st.session_state.doc_text)

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

            st.text_area("Raw text", value=st.session_state.doc_text,
                         height=400, label_visibility="collapsed")

            st.markdown("</div>", unsafe_allow_html=True)
    
# ── Upload area — full width ───────────────────────────────────

if not st.session_state.document_uploaded:
    st.markdown(f'<div style="{pad} padding-bottom:0.5rem;">', unsafe_allow_html=True)

    st.markdown("""<div style="font-family:'DM Mono',monospace; font-size:0.62rem;
                color:#6B6560; text-transform:uppercase; letter-spacing:0.1em;
                margin-bottom:0.5rem;text-align: center;">Upload Document</div>""", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        div[data-testid="stFileUploader"] {
            align-items: center;
            flex-direction: column;
            justify-content: center;
            display:flex;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    uploaded = st.file_uploader("Upload", type=["pdf", "txt"], label_visibility="collapsed")
    if uploaded:
        col_left, col_btn, col_right = st.columns([1, 1.2, 1])

        with col_btn:
            if st.button("⚖ Process Document", type="primary", use_container_width=True):
                with st.spinner("Extracting and indexing..."):
                    raw = uploaded.read()
                    text = extract_text(raw, uploaded.name)
                    doc_id = uploaded.name.replace(" ", "_").replace(".", "_")
                    col = index_document(text, doc_id)
                    st.session_state.doc_text = text
                    st.session_state.doc_id = doc_id
                    st.session_state.collection = col
                    st.session_state.chat_history = []
                    st.session_state.analysis_cache = {}
                st.success(f"Ready — {len(text):,} characters indexed")
                time.sleep(5)
                st.session_state.document_uploaded = True
                st.rerun()

# ── Footer ────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:1.5rem 3rem;border-top:1px solid #D4CCB8;margin-top:2rem;background:#EDE8DC;">

<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:3rem;margin-bottom:1rem;">
<span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#3A342F;">
VerdictAI · Legal Intelligence · Globally applicable
</span>

<span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#3A342F;text-align:right;">
Not legal advice · Consult a qualified attorney for important decisions
</span>
</div>

<div style="display:flex;justify-content:space-between;gap:2rem;flex-wrap:wrap;padding-top:1rem;border-top:1px solid #D4CCB8;">

<div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#5A534D;line-height:1.8;max-width:250px;">
<div style="color:#C9A84C;font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;">
About
</div>
VerdictAI analyzes legal documents and answers questions in plain English using retrieval-augmented AI and clause-aware document search.
</div>

<div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#5A534D;line-height:1.8;max-width:250px;">
<div style="color:#C9A84C;font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;">
Supported
</div>
Contracts · NDAs · Leases<br>
Employment Agreements<br>
Terms of Service · Court Filings
</div>

<div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#5A534D;line-height:1.8;max-width:250px;">
<div style="color:#C9A84C;font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;">
Powered By
</div>
Model · {OLLAMA_MODEL}<br>
Ollama Cloud · ChromaDB<br>
HuggingFace Embeddings<br>
Whisper · RAG Pipeline
</div>

<div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#5A534D;line-height:1.8;max-width:250px;">
<div style="color:#C9A84C;font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;">
Disclaimer
</div>
AI-generated legal information only. Always review important legal matters with a qualified attorney.
</div>

</div>

</div>
""", unsafe_allow_html=True)
