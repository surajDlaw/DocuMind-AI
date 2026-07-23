import os

import fitz
import faiss
import streamlit as st

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    .stAppDeployButton {display:none;}

    /* Keep Streamlit's sidebar collapse/reopen control visible. */
    [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 2. ENVIRONMENT VARIABLES
# =========================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit Cloud / production deployment fallback.
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = None

if not GROQ_API_KEY:
    st.error(
        "Groq API key not found. Add GROQ_API_KEY to your .env file."
    )
    st.stop()


# =========================================================
# 3. GROQ CLIENT
# =========================================================
client = Groq(api_key=GROQ_API_KEY)


# =========================================================
# 4. EMBEDDING MODEL
# =========================================================
@st.cache_resource
def load_embedding_model():
    """
    Load and cache the embedding model.
    """
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# =========================================================
# 5. PDF TEXT EXTRACTION
# =========================================================
def extract_pdf_text(uploaded_file):
    """
    Extract text from each readable PDF page.

    Keeps:
    - Document name
    - Page number
    - Extracted text
    """

    pages_data = []

    try:

        pdf_bytes = uploaded_file.getvalue()

        pdf_document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        for page_number, page in enumerate(
            pdf_document
        ):

            text = page.get_text(
                "text"
            ).strip()

            if text:

                pages_data.append({
                    "document": uploaded_file.name,
                    "page": page_number + 1,
                    "text": text
                })

        pdf_document.close()

    except Exception as e:

        st.error(
            f"Could not read {uploaded_file.name}: {e}"
        )

    return pages_data



# =========================================================
# 5B. PDF PROCESSING DIAGNOSTICS
# =========================================================
def inspect_pdf(uploaded_file):
    """
    Inspect a PDF for UI reporting without changing the document
    extraction or AI pipeline.
    """
    report = {
        "name": uploaded_file.name,
        "status": "Not Processed",
        "total_pages": 0,
        "readable_pages": 0,
        "reason": "",
        "recommendation": ""
    }

    pdf_document = None

    try:
        pdf_document = fitz.open(
            stream=uploaded_file.getvalue(),
            filetype="pdf"
        )

        report["total_pages"] = pdf_document.page_count

        if pdf_document.needs_pass:
            report["reason"] = "Password-protected or encrypted PDF."
            report["recommendation"] = (
                "Remove the password/encryption and upload the PDF again."
            )
            return report

        if pdf_document.page_count == 0:
            report["reason"] = "The PDF contains no pages."
            report["recommendation"] = "Upload a valid PDF containing document pages."
            return report

        readable_pages = 0

        for page in pdf_document:
            if page.get_text("text").strip():
                readable_pages += 1

        report["readable_pages"] = readable_pages

        if readable_pages == 0:
            report["reason"] = "No extractable text was detected."
            report["recommendation"] = (
                "This is likely a scanned/image-based PDF. "
                "Run OCR or upload a text-based PDF."
            )
        elif readable_pages < pdf_document.page_count:
            report["status"] = "Ready"
            report["reason"] = (
                f"Partially readable: {readable_pages} of "
                f"{pdf_document.page_count} page(s) contain extractable text."
            )
            report["recommendation"] = (
                "OCR may be required for the remaining page(s)."
            )
        else:
            report["status"] = "Ready"
            report["reason"] = "Text extracted successfully."

    except fitz.FileDataError:
        report["reason"] = "Invalid, damaged, or corrupted PDF."
        report["recommendation"] = (
            "Open and re-save/repair the PDF, then upload it again."
        )
    except Exception as e:
        report["reason"] = f"PDF processing error: {type(e).__name__}."
        report["recommendation"] = (
            "Verify that the file is a valid accessible PDF and try again."
        )
    finally:
        if pdf_document is not None:
            try:
                pdf_document.close()
            except Exception:
                pass

    return report


# =========================================================
# 6. TEXT CHUNKING
# =========================================================
def create_chunks(
    pages_data,
    chunk_size=1200,
    overlap=200
):
    """
    Split PDF text into overlapping chunks.

    Each chunk preserves:
    - Document name
    - Page number
    - Chunk number
    """

    chunks = []

    for page in pages_data:

        text = page["text"].strip()

        if not text:
            continue

        start = 0
        chunk_number = 1

        while start < len(text):

            end = min(
                start + chunk_size,
                len(text)
            )

            chunk_text = text[
                start:end
            ].strip()

            if chunk_text:

                chunks.append({
                    "document": page["document"],
                    "page": page["page"],
                    "chunk": chunk_number,
                    "text": chunk_text
                })

            if end >= len(text):
                break

            start += (
                chunk_size - overlap
            )

            chunk_number += 1

    return chunks


# =========================================================
# 7. CREATE EMBEDDINGS + FAISS INDEX
# =========================================================
def create_vector_index(chunks):
    """
    Convert chunks into embeddings and
    store them in a FAISS vector index.
    """

    if not chunks:
        return None

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    # Normalized embeddings + inner product
    # works as cosine-style similarity.
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# =========================================================
# 8. SEMANTIC RETRIEVAL + RELEVANCE FILTERING
# =========================================================

def _normalize_query_text(text):
    """Normalize text for exact-term retrieval while preserving semantic search."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9.%/+_-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _query_terms(text):
    stopwords = {
        "the","a","an","is","are","was","were","what","which","who","how",
        "do","does","did","i","me","my","of","in","on","for","to","from",
        "and","or","with","this","that","these","those","please","tell"
    }
    return [
        token for token in _normalize_query_text(text).split()
        if len(token) > 1 and token not in stopwords
    ]


def _lexical_score(question, text):
    """
    Exact/keyword score for facts semantic embeddings can miss:
    semester numbers, marks, IDs, dates, names and technical terms.
    """
    q_terms = _query_terms(question)
    if not q_terms:
        return 0.0

    normalized_text = _normalize_query_text(text)
    text_tokens = set(normalized_text.split())

    matched = sum(1 for term in q_terms if term in text_tokens)
    coverage = matched / max(len(q_terms), 1)

    # Reward exact numeric/identifier matches.
    q_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", question.lower()))
    number_bonus = 0.0
    if q_numbers:
        present = sum(1 for n in q_numbers if n in normalized_text)
        number_bonus = present / len(q_numbers)

    # Reward a literal phrase when present.
    normalized_question = _normalize_query_text(question)
    phrase_bonus = 1.0 if normalized_question and normalized_question in normalized_text else 0.0

    return min(1.0, 0.65 * coverage + 0.25 * number_bonus + 0.10 * phrase_bonus)


def retrieve_relevant_chunks(
    question,
    index,
    chunks,
    top_k=4,
    similarity_threshold=0.30
):
    """
    Hybrid retrieval:
    - semantic similarity from MiniLM + FAISS
    - lexical/exact-term matching for numbers, names and identifiers
    - candidate reranking before context reaches the LLM
    """

    if index is None or not chunks:
        return []

    # Retrieve a broader semantic candidate pool first.
    candidate_k = min(max(top_k * 4, 12), len(chunks))

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        question_embedding,
        candidate_k
    )

    semantic_by_index = {}
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            semantic_by_index[int(idx)] = float(score)

    # Add strong lexical candidates even when FAISS misses them.
    lexical_ranked = []
    for idx, chunk in enumerate(chunks):
        lexical = _lexical_score(question, chunk["text"])
        if lexical > 0:
            lexical_ranked.append((lexical, idx))

    lexical_ranked.sort(reverse=True)
    candidate_indices = set(semantic_by_index)
    candidate_indices.update(idx for _, idx in lexical_ranked[:candidate_k])

    ranked = []

    for idx in candidate_indices:
        chunk = chunks[idx]
        semantic = semantic_by_index.get(idx, 0.0)
        lexical = _lexical_score(question, chunk["text"])

        # Semantic remains primary; lexical evidence rescues exact-fact queries.
        hybrid = (0.72 * max(semantic, 0.0)) + (0.28 * lexical)

        ranked.append(
            {
                **chunk,
                "score": semantic,
                "semantic_score": semantic,
                "lexical_score": lexical,
                "hybrid_score": hybrid
            }
        )

    ranked.sort(key=lambda item: item["hybrid_score"], reverse=True)

    # Keep evidence if either semantic retrieval or exact-term evidence is useful.
    filtered = [
        item for item in ranked
        if item["semantic_score"] >= similarity_threshold
        or item["lexical_score"] >= 0.34
    ]

    return filtered[:top_k]


# =========================================================
# 9. LLM FUNCTION
# =========================================================
def ask_llm(prompt):
    """
    Send a prompt to Llama through Groq.
    """

    try:

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are DocuMind AI, a professional "
                        "multi-document analysis assistant. "
                        "Use only the document information provided "
                        "to you. Never invent facts. "
                        "If sufficient evidence is unavailable, "
                        "clearly state that the uploaded documents "
                        "do not contain enough information."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2
        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as e:

        st.error(
            f"LLM Error: {e}"
        )

        return None


# =========================================================
# 10. MULTI-DOCUMENT COMPARISON
# =========================================================
def compare_documents(all_documents):
    """
    Generate a professional comparative analysis
    across multiple uploaded documents.
    """

    grouped_documents = {}

    for page in all_documents:

        document_name = page[
            "document"
        ]

        if document_name not in grouped_documents:

            grouped_documents[
                document_name
            ] = []

        grouped_documents[
            document_name
        ].append(
            page["text"]
        )

    # Need at least two documents
    if len(grouped_documents) < 2:
        return None

    comparison_context = ""

    for document_name, pages in (
        grouped_documents.items()
    ):

        document_text = "\n".join(
            pages
        )

        comparison_context += f"""

========================================
DOCUMENT: {document_name}
========================================

{document_text}

"""

    comparison_prompt = f"""
You are performing a professional comparative analysis
of multiple uploaded documents.

Use ONLY the supplied document content.

DOCUMENTS:

{comparison_context}


Create the following analysis:


## 1. Document Overview

Briefly explain the purpose and main topic of each document.


## 2. Comparison Table

Create a clear Markdown table comparing all documents.

Use these dimensions:

- Main Topic
- Objective
- Approach / Methodology
- Key Findings
- Benefits / Strengths
- Limitations
- Conclusion

If information for a particular dimension is unavailable,
write "Not specified".


## 3. Key Similarities

Identify and explain the important similarities between
the documents.


## 4. Key Differences

Identify and explain their important differences.


## 5. Methodology Comparison

Compare the methodologies or approaches used.

If methodology information is unavailable, clearly
state that.


## 6. Findings Comparison

Compare the important findings and results.


## 7. Limitations Comparison

Compare the limitations identified by each document.


## 8. Overall Comparative Analysis

Provide a concise professional conclusion explaining
how the documents relate to one another.


IMPORTANT RULES:

- Use ONLY information contained in the supplied documents.
- Do not use outside knowledge.
- Never invent facts.
- Never invent methodologies, findings or limitations.
- Do not claim one document is better unless the supplied
  evidence directly supports that conclusion.
- Keep the comparison objective.
- Clearly distinguish information belonging to different
  documents.
"""

    return ask_llm(
        comparison_prompt
    )


# =========================================================
# 11. DOCUMIND AI — PROFESSIONAL PRODUCT UI
# =========================================================
st.markdown("""
<style>
/* ---------- App ---------- */
.stApp {
    background:
        radial-gradient(circle at 75% -10%, rgba(37,99,235,.09), transparent 27rem),
        #070b12;
}
.block-container {
    max-width: 1240px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: #0a101b;
    border-right: 1px solid #1c2635;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}
.sidebar-brand {
    padding: .55rem .2rem 1.25rem;
    border-bottom: 1px solid #1c2635;
    margin-bottom: 1.25rem;
}
.sidebar-logo {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -.02em;
}
.sidebar-sub {
    margin-top: .28rem;
    color: #7f8da3;
    font-size: .78rem;
}
.sidebar-label {
    color: #718096;
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: .11em;
    margin: .8rem 0 .5rem;
}
.stack-item {
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:.55rem .7rem;
    margin:.35rem 0;
    background:#0e1725;
    border:1px solid #1b293c;
    border-radius:9px;
    font-size:.78rem;
}
.stack-item span:last-child { color:#9db2cf; }

/* ---------- Hero ---------- */
.hero {
    position: relative;
    overflow: hidden;
    padding: 2.5rem 2.65rem;
    border: 1px solid #1e2c40;
    border-radius: 20px;
    background:
        radial-gradient(circle at 88% 15%, rgba(59,130,246,.16), transparent 22rem),
        linear-gradient(135deg,#0d1727 0%,#0a1220 55%,#0b1424 100%);
    box-shadow: 0 18px 55px rgba(0,0,0,.20);
    margin-bottom: 1.5rem;
}
.eyebrow {
    display:inline-flex;
    padding:.32rem .65rem;
    border:1px solid #294a78;
    border-radius:999px;
    color:#76a9fa;
    background:rgba(37,99,235,.08);
    font-size:.67rem;
    font-weight:800;
    letter-spacing:.09em;
}
.hero h1 {
    margin:.9rem 0 .35rem;
    font-size:2.65rem;
    line-height:1.08;
    letter-spacing:-.045em;
}
.hero h2 {
    margin:0 0 .85rem;
    font-size:1.22rem;
    font-weight:600;
    color:#d7e1ee;
}
.hero p {
    max-width:760px;
    color:#93a4ba;
    line-height:1.7;
    margin:0;
}
.tech-pills {
    display:flex;
    flex-wrap:wrap;
    gap:.45rem;
    margin-top:1.3rem;
}
.tech-pill {
    padding:.35rem .65rem;
    border-radius:8px;
    border:1px solid #22324a;
    background:#0b1320;
    color:#9eb0c8;
    font-size:.72rem;
}

/* ---------- Section ---------- */
.section-head {
    display:flex;
    align-items:flex-end;
    justify-content:space-between;
    gap:1rem;
    margin:2rem 0 1rem;
}
.section-title {
    font-size:1.35rem;
    font-weight:800;
    letter-spacing:-.025em;
}
.section-sub {
    color:#7f8da3;
    font-size:.82rem;
    margin-top:.2rem;
}

/* ---------- Cards ---------- */
.feature-card {
    min-height: 178px;
    padding:1.35rem;
    border:1px solid #1b293b;
    border-radius:15px;
    background:linear-gradient(180deg,#0d1624,#0b1320);
    box-shadow:0 10px 30px rgba(0,0,0,.12);
}
.feature-icon {
    width:34px;
    height:34px;
    display:flex;
    align-items:center;
    justify-content:center;
    border-radius:9px;
    background:#13233a;
    border:1px solid #24456f;
    margin-bottom:.9rem;
    font-size:1rem;
}
.feature-card h3 {
    margin:0 0 .45rem;
    font-size:.98rem;
}
.feature-card p {
    margin:0;
    color:#8291a6;
    font-size:.82rem;
    line-height:1.55;
}

/* ---------- Metrics ---------- */
div[data-testid="stMetric"] {
    background:linear-gradient(180deg,#0d1624,#0b1320);
    border:1px solid #1b293b;
    border-radius:13px;
    padding:.9rem 1rem;
}
div[data-testid="stMetricLabel"] { color:#8291a6; }
div[data-testid="stMetricValue"] { font-size:1.65rem; }

/* ---------- Streamlit controls ---------- */
.stButton > button {
    border-radius:9px;
    min-height:2.55rem;
    font-weight:700;
}
div[data-testid="stFileUploader"] {
    border-radius:12px;
}
div[data-testid="stExpander"] {
    border:1px solid #1b293b;
    border-radius:12px;
    background:#0a111d;
}
div[data-baseweb="tab-list"] {
    gap:.35rem;
    background:#0a111d;
    border:1px solid #1b293b;
    padding:.35rem;
    border-radius:12px;
}
button[data-baseweb="tab"] {
    border-radius:8px;
    padding:.55rem 1rem;
    font-weight:750;
}
hr { border-color:#1c2635 !important; }

/* ---------- Status ---------- */
.status-note {
    padding:.85rem 1rem;
    border:1px solid #1b293b;
    border-radius:11px;
    background:#0b1421;
    color:#91a1b7;
    font-size:.82rem;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 12. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">📚 DocuMind AI</div>
        <div class="sidebar-sub">Multi-Document Intelligence</div>
    </div>
    <div class="sidebar-label">DOCUMENTS</div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload one or more PDF documents."
    )

    if uploaded_files:
        total_upload_mb = sum(len(f.getvalue()) for f in uploaded_files) / (1024 * 1024)
        st.caption(f"{len(uploaded_files)} file(s) • {total_upload_mb:.1f} MB")
        oversized = [f.name for f in uploaded_files if len(f.getvalue()) > 25 * 1024 * 1024]
        if oversized:
            st.error("These PDFs exceed the 25 MB per-file limit: " + ", ".join(oversized))
            st.stop()
        if total_upload_mb > 75:
            st.error("Upload is too large for reliable analysis. Keep the combined PDF size under 75 MB.")
            st.stop()

    st.markdown("""
    <div class="sidebar-label">AI ENGINE</div>
    <div class="stack-item"><span>Language Model</span><span>Llama 3.3 70B</span></div>
    <div class="stack-item"><span>Embeddings</span><span>MiniLM</span></div>
    <div class="stack-item"><span>Vector Search</span><span>FAISS</span></div>
    <div class="stack-item"><span>Pipeline</span><span>RAG</span></div>
    """, unsafe_allow_html=True)


# =========================================================
# 13. HERO
# =========================================================
st.markdown("""
<div class="hero">
    <span class="eyebrow">AI DOCUMENT INTELLIGENCE</span>
    <h1>DocuMind AI</h1>
    <h2>Understand multiple documents. Faster.</h2>
    <p>
        Upload PDFs, generate structured summaries, compare selected documents,
        and ask evidence-grounded questions through a retrieval-augmented AI workflow.
    </p>
    <div class="tech-pills">
        <span class="tech-pill">Semantic Search</span>
        <span class="tech-pill">MiniLM Embeddings</span>
        <span class="tech-pill">FAISS Retrieval</span>
        <span class="tech-pill">Llama 3.3 70B</span>
        <span class="tech-pill">Evidence Grounding</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 14. EMPTY STATE
# =========================================================
if not uploaded_files:
    st.markdown("""
    <div class="section-head">
        <div>
            <div class="section-title">Document Intelligence</div>
            <div class="section-sub">Three focused workflows for working with your PDFs.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">▤</div>
            <h3>AI Summary</h3>
            <p>Turn long PDFs into structured reports with topics, findings, methodology, limitations and conclusions.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⇄</div>
            <h3>Compare Documents</h3>
            <p>Choose any two or more readable PDFs and compare only the files you want across consistent dimensions.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⌕</div>
            <h3>Ask DocuMind</h3>
            <p>Ask questions against your documents and inspect the exact retrieved evidence used for the response.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-head">
        <div>
            <div class="section-title">How it works</div>
            <div class="section-sub">A retrieval-first workflow designed to keep answers grounded in your files.</div>
        </div>
    </div>
    <div class="status-note">
        PDF Upload &nbsp; → &nbsp; Text Extraction &nbsp; → &nbsp; Chunking &nbsp; → &nbsp;
        Embeddings &nbsp; → &nbsp; FAISS Retrieval &nbsp; → &nbsp; AI Response
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# =========================================================
# 15. PROCESS DOCUMENTS
# =========================================================
all_documents = []
processing_reports = []

for file in uploaded_files:
    report = inspect_pdf(file)
    processing_reports.append(report)

    if report["status"] == "Ready":
        all_documents.extend(extract_pdf_text(file))

ready_reports = [r for r in processing_reports if r["status"] == "Ready"]
failed_reports = [r for r in processing_reports if r["status"] != "Ready"]


# =========================================================
# 16. PROCESSING OVERVIEW
# =========================================================
st.markdown("""
<div class="section-head">
    <div>
        <div class="section-title">Document Processing</div>
        <div class="section-sub">Validation and extraction status for the current upload.</div>
    </div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Uploaded", len(uploaded_files))
m2.metric("Ready", len(ready_reports))
m3.metric("Need Attention", len(failed_reports))
m4.metric("Readable Pages", len(all_documents))

if failed_reports:
    st.warning(
        f"{len(ready_reports)} of {len(uploaded_files)} PDF(s) are ready. "
        f"{len(failed_reports)} require attention."
    )
else:
    st.success(f"All {len(uploaded_files)} uploaded PDF(s) are ready for analysis.")

with st.expander("Document health report", expanded=bool(failed_reports)):
    for i, report in enumerate(processing_reports, 1):
        if report["status"] == "Ready":
            st.success(
                f"{report['name']}  •  Ready  •  "
                f"{report['readable_pages']}/{report['total_pages']} readable page(s)"
            )
        else:
            st.error(f"{report['name']}  •  Not processed")

        st.markdown(f"**Reason:** {report['reason']}")
        if report["recommendation"]:
            st.caption(f"Recommended action: {report['recommendation']}")
        if i < len(processing_reports):
            st.divider()

if not all_documents:
    st.error("No readable document content is available. Review the document health report.")
    st.stop()


# =========================================================
# 17. RAG INDEX
# =========================================================
document_chunks = create_chunks(all_documents)

with st.spinner("Building semantic document index..."):
    vector_index = create_vector_index(document_chunks)

document_names = list(dict.fromkeys(p["document"] for p in all_documents))

with st.expander("Technical details", expanded=False):
    d1, d2, d3 = st.columns(3)
    d1.metric("Readable Pages", len(all_documents))
    d2.metric("RAG Chunks", len(document_chunks))
    d3.metric("Indexed Chunks", vector_index.ntotal if vector_index is not None else 0)

    preview_tab, chunks_tab = st.tabs(["Extracted Text", "RAG Chunks"])

    with preview_tab:
        for page in all_documents[:5]:
            st.markdown(f"**{page['document']} — Page {page['page']}**")
            st.text_area(
                "Extracted text",
                value=page["text"][:2000],
                height=170,
                disabled=True,
                key=f"preview_{page['document']}_{page['page']}",
                label_visibility="collapsed"
            )

    with chunks_tab:
        for chunk in document_chunks[:5]:
            st.markdown(
                f"**{chunk['document']} — Page {chunk['page']} — Chunk {chunk['chunk']}**"
            )
            st.text_area(
                "Chunk text",
                value=chunk["text"],
                height=150,
                disabled=True,
                key=f"chunk_{chunk['document']}_{chunk['page']}_{chunk['chunk']}",
                label_visibility="collapsed"
            )


# =========================================================
# 18. DOCUMENT INTELLIGENCE WORKSPACE
# =========================================================
st.markdown("""
<div class="section-head">
    <div>
        <div class="section-title">Document Intelligence Workspace</div>
        <div class="section-sub">Summarize, compare or query your processed documents.</div>
    </div>
</div>
""", unsafe_allow_html=True)

summary_tab, compare_tab, ask_tab = st.tabs(
    ["▤  Summary", "⇄  Compare", "⌕  Ask AI"]
)


# ---------------- SUMMARY ----------------
with summary_tab:
    st.subheader("AI Document Summary")
    st.caption("Generate a structured professional report for every readable uploaded document.")

    if st.button("Generate Summaries", type="primary", key="summarize_documents"):
        document_text = "".join(
            f"\n\nDOCUMENT: {p['document']}\nPAGE: {p['page']}\nCONTENT:\n{p['text']}"
            for p in all_documents
        )

        summary_prompt = f"""
Analyze the uploaded documents and create a separate professional summary for each document.

For every document provide:
1. Document Name
2. Main Topic
3. Key Points
4. Methodology (if available)
5. Important Results or Findings
6. Limitations (if available)
7. Conclusion

Rules:
- Use only the supplied document content.
- Never invent missing information.
- Clearly separate each document.
- Keep the analysis structured and professional.

DOCUMENT CONTENT:
{document_text}
"""
        with st.spinner("Analyzing documents..."):
            summary = ask_llm(summary_prompt)

        if summary:
            st.markdown("### Analysis")
            st.markdown(summary)


# ---------------- COMPARE ----------------
with compare_tab:
    st.subheader("Multi-Document Comparison")
    st.caption("Select any two or more readable documents. Only your selected files are compared.")

    if len(document_names) >= 2:
        selected_documents = st.multiselect(
            "Documents to compare",
            options=document_names,
            default=document_names[:2],
            key="comparison_document_selector",
            help="Select 2 or more documents."
        )

        st.caption(
            f"{len(selected_documents)} selected from {len(document_names)} readable document(s)."
        )

        if st.button("Compare Selected Documents", type="primary", key="compare_documents"):
            if len(selected_documents) < 2:
                st.warning("Select at least two documents to run a comparison.")
            else:
                selected_pages = [
                    p for p in all_documents
                    if p["document"] in selected_documents
                ]

                with st.spinner(f"Comparing {len(selected_documents)} documents..."):
                    comparison = compare_documents(selected_pages)

                if comparison:
                    st.markdown("### Comparative Analysis")
                    st.caption("Compared: " + " • ".join(selected_documents))
                    st.markdown(comparison)
    else:
        st.info("At least two readable PDFs are required for comparison.")


# ---------------- ASK AI ----------------
with ask_tab:
    st.subheader("Ask Your Documents")
    st.caption("DocuMind retrieves relevant evidence first, then generates a grounded answer.")

    question = st.text_input(
        "Question",
        placeholder="Example: What are the main findings and limitations across these documents?",
        key="document_question"
    )

    if st.button("Ask DocuMind", type="primary", key="ask_documind"):
        if not question.strip():
            st.warning("Enter a question first.")
        elif vector_index is None:
            st.error("The semantic search index is unavailable.")
        else:
            with st.spinner("Searching documents and generating a grounded answer..."):
                relevant_chunks = retrieve_relevant_chunks(
                    question=question,
                    index=vector_index,
                    chunks=document_chunks,
                    top_k=4,
                    similarity_threshold=0.30
                )

                if not relevant_chunks:
                    st.warning("No sufficiently relevant evidence was found in the uploaded documents.")
                else:
                    context = "\n\n".join(
                        f"DOCUMENT: {c['document']}\n"
                        f"PAGE: {c['page']}\n"
                        f"CHUNK: {c['chunk']}\n"
                        f"CONTENT:\n{c['text']}"
                        for c in relevant_chunks
                    )

                    rag_prompt = f"""
Answer the user's question using ONLY the retrieved document context below.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

RULES:
1. Use only retrieved document evidence.
2. Do not use outside knowledge.
3. Never invent facts.
4. If evidence is insufficient, state that the uploaded documents do not contain enough information.
5. Give a clear and professional answer.
6. Do not invent document names or page numbers.
7. Distinguish which document supports a claim when useful.
"""

                    answer = ask_llm(rag_prompt)

                    if answer:
                        st.markdown("### Answer")
                        st.markdown(answer)

                        st.markdown("### Sources")
                        seen = set()
                        for chunk in relevant_chunks:
                            source = (chunk["document"], chunk["page"])
                            if source not in seen:
                                seen.add(source)
                                st.markdown(
                                    f"**{chunk['document']}**  ·  Page {chunk['page']}"
                                )

                        with st.expander("Retrieved evidence", expanded=False):
                            for n, chunk in enumerate(relevant_chunks, 1):
                                st.markdown(
                                    f"**Evidence {n} — {chunk['document']} — "
                                    f"Page {chunk['page']} — Chunk {chunk['chunk']}**"
                                )
                                st.caption(f"Hybrid relevance: {chunk.get('hybrid_score', chunk['score']):.3f}  •  Semantic: {chunk.get('semantic_score', chunk['score']):.3f}  •  Exact-term match: {chunk.get('lexical_score', 0.0):.3f}")
                                st.text_area(
                                    "Evidence",
                                    value=chunk["text"],
                                    height=190,
                                    disabled=True,
                                    key=(
                                        f"evidence_{n}_{chunk['document']}_"
                                        f"{chunk['page']}_{chunk['chunk']}"
                                    ),
                                    label_visibility="collapsed"
                                )

st.divider()
st.caption("DocuMind AI • Multi-Document Intelligence • Evidence-grounded document analysis.")
