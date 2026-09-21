import os
import re
from typing import Dict, List, Optional, Tuple

import faiss
import fitz
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
    layout="wide",
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    .stAppDeployButton {display: none;}

    [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

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

    [data-testid="stSidebar"] {
        background: #0a101b;
        border-right: 1px solid #1c2635;
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
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: .55rem .7rem;
        margin: .35rem 0;
        background: #0e1725;
        border: 1px solid #1b293c;
        border-radius: 9px;
        font-size: .78rem;
    }

    .stack-item span:last-child {
        color: #9db2cf;
    }

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
        display: inline-flex;
        padding: .32rem .65rem;
        border: 1px solid #294a78;
        border-radius: 999px;
        color: #76a9fa;
        background: rgba(37,99,235,.08);
        font-size: .67rem;
        font-weight: 800;
        letter-spacing: .09em;
    }

    .hero h1 {
        margin: .9rem 0 .35rem;
        font-size: 2.65rem;
        line-height: 1.08;
        letter-spacing: -.045em;
    }

    .hero h2 {
        margin: 0 0 .85rem;
        font-size: 1.22rem;
        font-weight: 600;
        color: #d7e1ee;
    }

    .hero p {
        max-width: 760px;
        color: #93a4ba;
        line-height: 1.7;
        margin: 0;
    }

    .tech-pills {
        display: flex;
        flex-wrap: wrap;
        gap: .45rem;
        margin-top: 1.3rem;
    }

    .tech-pill {
        padding: .35rem .65rem;
        border-radius: 8px;
        border: 1px solid #22324a;
        background: #0b1320;
        color: #9eb0c8;
        font-size: .72rem;
    }

    .section-head {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1rem;
        margin: 2rem 0 1rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -.025em;
    }

    .section-sub {
        color: #7f8da3;
        font-size: .82rem;
        margin-top: .2rem;
    }

    .feature-card {
        min-height: 178px;
        padding: 1.35rem;
        border: 1px solid #1b293b;
        border-radius: 15px;
        background: linear-gradient(180deg,#0d1624,#0b1320);
        box-shadow: 0 10px 30px rgba(0,0,0,.12);
    }

    .feature-icon {
        width: 34px;
        height: 34px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 9px;
        background: #13233a;
        border: 1px solid #24456f;
        margin-bottom: .9rem;
        font-size: 1rem;
    }

    .feature-card h3 {
        margin: 0 0 .45rem;
        font-size: .98rem;
    }

    .feature-card p {
        margin: 0;
        color: #8291a6;
        font-size: .82rem;
        line-height: 1.55;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg,#0d1624,#0b1320);
        border: 1px solid #1b293b;
        border-radius: 13px;
        padding: .9rem 1rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #8291a6;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.65rem;
    }

    .stButton > button {
        border-radius: 9px;
        min-height: 2.55rem;
        font-weight: 700;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #1b293b;
        border-radius: 12px;
        background: #0a111d;
    }

    div[data-baseweb="tab-list"] {
        gap: .35rem;
        background: #0a111d;
        border: 1px solid #1b293b;
        padding: .35rem;
        border-radius: 12px;
    }

    button[data-baseweb="tab"] {
        border-radius: 8px;
        padding: .55rem 1rem;
        font-weight: 750;
    }

    hr {
        border-color: #1c2635 !important;
    }

    .status-note {
        padding: .85rem 1rem;
        border: 1px solid #1b293b;
        border-radius: 11px;
        background: #0b1421;
        color: #91a1b7;
        font-size: .82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 2. CONFIGURATION
# =========================================================

MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

MAX_FILE_MB = 25
MAX_TOTAL_UPLOAD_MB = 75
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


# =========================================================
# 3. API KEY + GROQ CLIENT
# =========================================================

load_dotenv()


def get_groq_api_key() -> Optional[str]:
    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        return api_key.strip()

    try:
        secret_key = st.secrets.get("GROQ_API_KEY")
        if secret_key:
            return str(secret_key).strip()
    except Exception:
        pass

    return None


GROQ_API_KEY = get_groq_api_key()

if not GROQ_API_KEY:
    st.error(
        "Groq API key not found. Add GROQ_API_KEY to Streamlit Secrets "
        "or your local .env file."
    )
    st.stop()


@st.cache_resource
def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


client = get_groq_client(GROQ_API_KEY)


# =========================================================
# 4. EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


embedding_model = load_embedding_model()


# =========================================================
# 5. PDF EXTRACTION
# =========================================================

def inspect_pdf(uploaded_file) -> Dict:
    report = {
        "name": uploaded_file.name,
        "status": "Not Processed",
        "total_pages": 0,
        "readable_pages": 0,
        "reason": "",
        "recommendation": "",
    }

    pdf_document = None

    try:
        pdf_document = fitz.open(
            stream=uploaded_file.getvalue(),
            filetype="pdf",
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
            report["recommendation"] = "Upload a valid PDF."
            return report

        readable_pages = 0

        for page in pdf_document:
            if page.get_text("text").strip():
                readable_pages += 1

        report["readable_pages"] = readable_pages

        if readable_pages == 0:
            report["reason"] = "No extractable text was detected."
            report["recommendation"] = (
                "This is likely a scanned/image PDF. "
                "Use OCR and upload the OCR-enabled PDF."
            )
        elif readable_pages < pdf_document.page_count:
            report["status"] = "Ready"
            report["reason"] = (
                f"Partially readable: {readable_pages} of "
                f"{pdf_document.page_count} page(s) contain text."
            )
            report["recommendation"] = (
                "OCR may be required for the remaining pages."
            )
        else:
            report["status"] = "Ready"
            report["reason"] = "Text extracted successfully."

    except fitz.FileDataError:
        report["reason"] = "Invalid, damaged, or corrupted PDF."
        report["recommendation"] = (
            "Open and re-save the PDF, then upload it again."
        )
    except Exception as exc:
        report["reason"] = f"PDF processing error: {type(exc).__name__}."
        report["recommendation"] = "Verify that the PDF is valid."
    finally:
        if pdf_document is not None:
            try:
                pdf_document.close()
            except Exception:
                pass

    return report


def extract_pdf_text(uploaded_file) -> List[Dict]:
    pages_data = []
    pdf_document = None

    try:
        pdf_document = fitz.open(
            stream=uploaded_file.getvalue(),
            filetype="pdf",
        )

        for page_number, page in enumerate(pdf_document):
            text = page.get_text("text").strip()

            if text:
                pages_data.append(
                    {
                        "document": uploaded_file.name,
                        "page": page_number + 1,
                        "text": text,
                    }
                )

    except Exception as exc:
        st.error(
            f"Could not read {uploaded_file.name}: {exc}"
        )
    finally:
        if pdf_document is not None:
            try:
                pdf_document.close()
            except Exception:
                pass

    return pages_data


# =========================================================
# 6. CHUNKING
# =========================================================

def create_chunks(
    pages_data: List[Dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:

    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    chunks = []

    for page in pages_data:
        text = page["text"].strip()

        if not text:
            continue

        start = 0
        chunk_number = 1

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "document": page["document"],
                        "page": page["page"],
                        "chunk": chunk_number,
                        "text": chunk_text,
                    }
                )

            if end >= len(text):
                break

            start += chunk_size - overlap
            chunk_number += 1

    return chunks


# =========================================================
# 7. FAISS INDEX
# =========================================================

@st.cache_data(show_spinner=False)
def embed_texts(texts: Tuple[str, ...]):
    embeddings = embedding_model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.astype("float32")


def create_vector_index(chunks: List[Dict]):
    if not chunks:
        return None

    texts = tuple(chunk["text"] for chunk in chunks)
    embeddings = embed_texts(texts)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


# =========================================================
# 8. HYBRID RETRIEVAL
# =========================================================

def normalize_query_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9.%/+_-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def query_terms(text: str) -> List[str]:
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were",
        "what", "which", "who", "how", "do", "does", "did",
        "i", "me", "my", "of", "in", "on", "for", "to",
        "from", "and", "or", "with", "this", "that",
        "these", "those", "please", "tell",
    }

    return [
        token
        for token in normalize_query_text(text).split()
        if len(token) > 1 and token not in stopwords
    ]


def lexical_score(question: str, text: str) -> float:
    terms = query_terms(question)

    if not terms:
        return 0.0

    normalized_text = normalize_query_text(text)
    text_tokens = set(normalized_text.split())

    matched = sum(
        1 for term in terms if term in text_tokens
    )

    coverage = matched / max(len(terms), 1)

    question_numbers = set(
        re.findall(r"\b\d+(?:\.\d+)?\b", question.lower())
    )

    number_bonus = 0.0

    if question_numbers:
        present = sum(
            1
            for number in question_numbers
            if number in normalized_text
        )
        number_bonus = present / len(question_numbers)

    normalized_question = normalize_query_text(question)

    phrase_bonus = 0.0

    if normalized_question:
        if normalized_question in normalized_text:
            phrase_bonus = 1.0

    return min(
        1.0,
        0.65 * coverage
        + 0.25 * number_bonus
        + 0.10 * phrase_bonus,
    )


def document_intent_score(
    question: str,
    filename: str,
    document_text: str = "",
) -> float:

    query = normalize_query_text(question)
    name = normalize_query_text(filename)
    sample = normalize_query_text(document_text[:5000])

    score = 0.0

    question_terms = set(query_terms(question))
    filename_terms = set(query_terms(filename))

    if question_terms and filename_terms:
        score += 0.55 * (
            len(question_terms & filename_terms)
            / len(question_terms)
        )

    intent_aliases = {
        "resume": {
            "resume", "cv", "curriculum vitae",
            "skill", "skills", "experience",
            "project", "projects", "education",
        },
        "result": {
            "result", "marks", "mark", "score",
            "semester", "sem", "sgpa", "cgpa",
            "grade", "grades",
        },
        "certificate": {
            "certificate", "certification",
            "certified", "completion", "course",
        },
        "research": {
            "research", "paper", "methodology",
            "finding", "findings", "abstract",
            "limitation", "limitations",
        },
        "invoice": {
            "invoice", "bill", "amount",
            "payment", "total", "tax",
        },
        "report": {
            "report", "analysis", "summary",
            "finding", "findings",
        },
    }

    for document_type, aliases in intent_aliases.items():
        query_hits = sum(
            1
            for alias in aliases
            if alias in query
        )

        if not query_hits:
            continue

        if document_type in name or any(
            alias in name for alias in aliases
        ):
            score += min(
                1.0,
                0.35 + 0.10 * query_hits,
            )

        if document_type in sample or any(
            alias in sample for alias in aliases
        ):
            score += min(
                0.30,
                0.08 * query_hits,
            )

    return min(score, 1.25)


def retrieve_relevant_chunks(
    question: str,
    index,
    chunks: List[Dict],
    top_k: int = 5,
    similarity_threshold: float = 0.25,
) -> List[Dict]:

    if index is None or not chunks:
        return []

    candidate_k = min(
        max(top_k * 6, 20),
        len(chunks),
    )

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(
        question_embedding,
        candidate_k,
    )

    semantic_by_index = {}

    for score, index_value in zip(
        scores[0],
        indices[0],
    ):
        if index_value >= 0:
            semantic_by_index[int(index_value)] = float(score)

    document_samples = {}

    for chunk in chunks:
        document_name = chunk["document"]

        if document_name not in document_samples:
            document_samples[document_name] = ""

        if len(document_samples[document_name]) < 5000:
            document_samples[document_name] += (
                " " + chunk["text"]
            )

    document_scores = {
        filename: document_intent_score(
            question,
            filename,
            text,
        )
        for filename, text in document_samples.items()
    }

    lexical_ranked = []

    for index_value, chunk in enumerate(chunks):
        lexical = lexical_score(
            question,
            chunk["text"],
        )

        if lexical > 0:
            lexical_ranked.append(
                (lexical, index_value)
            )

    lexical_ranked.sort(reverse=True)

    candidate_indices = set(
        semantic_by_index.keys()
    )

    candidate_indices.update(
        index_value
        for _, index_value
        in lexical_ranked[:candidate_k]
    )

    if document_scores:
        best_document_score = max(
            document_scores.values()
        )

        if best_document_score >= 0.35:
            strong_documents = {
                name
                for name, score
                in document_scores.items()
                if score >= max(
                    0.35,
                    best_document_score - 0.15,
                )
            }

            for index_value, chunk in enumerate(chunks):
                if chunk["document"] in strong_documents:
                    candidate_indices.add(index_value)

    ranked = []

    for index_value in candidate_indices:
        chunk = chunks[index_value]

        semantic = semantic_by_index.get(
            index_value,
            0.0,
        )

        lexical = lexical_score(
            question,
            chunk["text"],
        )

        document_score = document_scores.get(
            chunk["document"],
            0.0,
        )

        hybrid = (
            0.55 * max(semantic, 0.0)
            + 0.25 * lexical
            + 0.20 * min(document_score, 1.0)
        )

        ranked.append(
            {
                **chunk,
                "score": semantic,
                "semantic_score": semantic,
                "lexical_score": lexical,
                "document_score": document_score,
                "hybrid_score": hybrid,
            }
        )

    ranked.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )

    filtered = [
        item
        for item in ranked
        if (
            item["semantic_score"] >= similarity_threshold
            or item["lexical_score"] >= 0.30
            or item["document_score"] >= 0.35
        )
    ]

    results = []
    per_document = {}

    for item in filtered:
        filename = item["document"]

        document_limit = (
            3
            if item["document_score"] >= 0.35
            else 2
        )

        if per_document.get(filename, 0) >= document_limit:
            continue

        results.append(item)

        per_document[filename] = (
            per_document.get(filename, 0) + 1
        )

        if len(results) >= top_k:
            break

    return results


# =========================================================
# 9. GROQ LLM
# =========================================================

def ask_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
) -> Optional[str]:

    default_system_prompt = (
        "You are DocuMind AI, a professional multi-document "
        "analysis assistant. Use only the document information "
        "provided in the user prompt. Never invent facts. "
        "If the evidence is insufficient, clearly say that "
        "the uploaded documents do not contain enough information."
    )

    if system_prompt is None:
        system_prompt = default_system_prompt

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
        )

        if not response.choices:
            st.error("Groq returned an empty response.")
            return None

        content = response.choices[0].message.content

        if not content:
            st.error("Groq returned empty content.")
            return None

        return content.strip()

    except Exception as exc:
        error_text = str(exc)

        if "model_not_found" in error_text.lower():
            st.error(
                "The configured Groq model is unavailable. "
                "The app is configured for "
                f"{MODEL_NAME}."
            )
        elif "authentication" in error_text.lower() or "401" in error_text:
            st.error(
                "Groq authentication failed. Check GROQ_API_KEY "
                "in Streamlit Secrets."
            )
        elif "429" in error_text or "rate" in error_text.lower():
            st.error(
                "Groq rate limit reached. Please wait and try again."
            )
        else:
            st.error(f"LLM Error: {exc}")

        return None


# =========================================================
# 10. TEXT LIMITING
# =========================================================

def limit_text(text: str, max_characters: int = 45000) -> str:
    if len(text) <= max_characters:
        return text

    return (
        text[:max_characters]
        + "\n\n[Document content truncated for this request.]"
    )


# =========================================================
# 11. SUMMARY
# =========================================================

def summarize_document(
    document_name: str,
    document_pages: List[Dict],
) -> Optional[str]:

    document_text = "\n\n".join(
        (
            f"DOCUMENT: {page['document']}\n"
            f"PAGE: {page['page']}\n"
            f"CONTENT:\n{page['text']}"
        )
        for page in document_pages
    )

    document_text = limit_text(
        document_text,
        45000,
    )

    prompt = f"""
Analyze the uploaded document below.

DOCUMENT:
{document_text}

Create a professional structured summary with:

## 1. Document Name
## 2. Main Topic
## 3. Key Points
## 4. Methodology
## 5. Important Results or Findings
## 6. Limitations
## 7. Conclusion

Rules:
- Use only the supplied document.
- Never invent missing information.
- If a section is not available, write "Not specified in the document."
- Keep the answer factual and concise.
"""

    return ask_llm(prompt)


# =========================================================
# 12. COMPARISON
# =========================================================

def compare_documents(
    all_documents: List[Dict],
) -> Optional[str]:

    grouped_documents = {}

    for page in all_documents:
        document_name = page["document"]

        if document_name not in grouped_documents:
            grouped_documents[document_name] = []

        grouped_documents[document_name].append(
            page["text"]
        )

    if len(grouped_documents) < 2:
        return None

    comparison_context = ""

    for document_name, pages in grouped_documents.items():
        document_text = limit_text(
            "\n".join(pages),
            30000,
        )

        comparison_context += (
            "\n========================================\n"
            f"DOCUMENT: {document_name}\n"
            "========================================\n"
            f"{document_text}\n"
        )

    comparison_context = limit_text(
        comparison_context,
        90000,
    )

    prompt = f"""
Perform an objective comparative analysis of the supplied documents.

DOCUMENTS:
{comparison_context}

Create:

## 1. Document Overview
Briefly explain the purpose and main topic of each document.

## 2. Comparison Table
Compare:
- Main Topic
- Objective
- Approach / Methodology
- Key Findings
- Benefits / Strengths
- Limitations
- Conclusion

## 3. Key Similarities

## 4. Key Differences

## 5. Methodology Comparison

## 6. Findings Comparison

## 7. Limitations Comparison

## 8. Overall Comparative Analysis

Rules:
- Use ONLY supplied document content.
- Never invent facts.
- Write "Not specified" when information is unavailable.
- Keep documents clearly separated.
- Keep the comparison objective.
"""

    return ask_llm(prompt)


# =========================================================
# 13. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">📚 DocuMind AI</div>
            <div class="sidebar-sub">Multi-Document Intelligence</div>
        </div>

        <div class="sidebar-label">DOCUMENTS</div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload one or more PDF documents.",
    )

    if uploaded_files:
        total_upload_mb = sum(
            len(file.getvalue())
            for file in uploaded_files
        ) / (1024 * 1024)

        st.caption(
            f"{len(uploaded_files)} file(s) • "
            f"{total_upload_mb:.1f} MB"
        )

        oversized = [
            file.name
            for file in uploaded_files
            if len(file.getvalue())
            > MAX_FILE_MB * 1024 * 1024
        ]

        if oversized:
            st.error(
                "These PDFs exceed the "
                f"{MAX_FILE_MB} MB per-file limit: "
                + ", ".join(oversized)
            )
            st.stop()

        if total_upload_mb > MAX_TOTAL_UPLOAD_MB:
            st.error(
                "Combined upload is too large. "
                f"Keep it under {MAX_TOTAL_UPLOAD_MB} MB."
            )
            st.stop()

    st.markdown(
        f"""
        <div class="sidebar-label">AI ENGINE</div>

        <div class="stack-item">
            <span>Language Model</span>
            <span>GPT-OSS 120B</span>
        </div>

        <div class="stack-item">
            <span>Embeddings</span>
            <span>MiniLM</span>
        </div>

        <div class="stack-item">
            <span>Vector Search</span>
            <span>FAISS</span>
        </div>

        <div class="stack-item">
            <span>Pipeline</span>
            <span>RAG</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 14. HERO
# =========================================================

st.markdown(
    """
    <div class="hero">
        <span class="eyebrow">AI DOCUMENT INTELLIGENCE</span>

        <h1>DocuMind AI</h1>

        <h2>Understand multiple documents. Faster.</h2>

        <p>
            Upload PDFs, generate structured summaries, compare
            selected documents, and ask evidence-grounded questions
            through a retrieval-augmented AI workflow.
        </p>

        <div class="tech-pills">
            <span class="tech-pill">Semantic Search</span>
            <span class="tech-pill">MiniLM Embeddings</span>
            <span class="tech-pill">FAISS Retrieval</span>
            <span class="tech-pill">GPT-OSS 120B</span>
            <span class="tech-pill">Evidence Grounding</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 15. EMPTY STATE
# =========================================================

if not uploaded_files:
    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">
                    Document Intelligence
                </div>

                <div class="section-sub">
                    Three focused workflows for working with your PDFs.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    column_one, column_two, column_three = st.columns(3)

    with column_one:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">▤</div>
                <h3>AI Summary</h3>
                <p>
                    Turn long PDFs into structured reports with
                    topics, findings, methodology, limitations
                    and conclusions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with column_two:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">⇄</div>
                <h3>Compare Documents</h3>
                <p>
                    Choose two or more readable PDFs and compare
                    only the files you want across consistent dimensions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with column_three:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">⌕</div>
                <h3>Ask DocuMind</h3>
                <p>
                    Ask questions against your documents and inspect
                    the retrieved evidence used for the answer.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="section-head">
            <div>
                <div class="section-title">How it works</div>
                <div class="section-sub">
                    A retrieval-first workflow designed to keep
                    answers grounded in your files.
                </div>
            </div>
        </div>

        <div class="status-note">
            PDF Upload → Text Extraction → Chunking →
            Embeddings → FAISS Retrieval → AI Response
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# =========================================================
# 16. PROCESS DOCUMENTS
# =========================================================

all_documents = []
processing_reports = []

for uploaded_file in uploaded_files:
    report = inspect_pdf(uploaded_file)
    processing_reports.append(report)

    if report["status"] == "Ready":
        all_documents.extend(
            extract_pdf_text(uploaded_file)
        )

ready_reports = [
    report
    for report in processing_reports
    if report["status"] == "Ready"
]

failed_reports = [
    report
    for report in processing_reports
    if report["status"] != "Ready"
]


# =========================================================
# 17. PROCESSING OVERVIEW
# =========================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-title">
                Document Processing
            </div>
            <div class="section-sub">
                Validation and extraction status for the current upload.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_one, metric_two, metric_three, metric_four = st.columns(4)

metric_one.metric(
    "Uploaded",
    len(uploaded_files),
)

metric_two.metric(
    "Ready",
    len(ready_reports),
)

metric_three.metric(
    "Need Attention",
    len(failed_reports),
)

metric_four.metric(
    "Readable Pages",
    len(all_documents),
)


if failed_reports:
    st.warning(
        f"{len(ready_reports)} of {len(uploaded_files)} "
        "PDF(s) are ready. "
        f"{len(failed_reports)} require attention."
    )
else:
    st.success(
        f"All {len(uploaded_files)} uploaded PDF(s) "
        "are ready for analysis."
    )


with st.expander(
    "Document health report",
    expanded=bool(failed_reports),
):
    for report_index, report in enumerate(
        processing_reports,
        start=1,
    ):
        if report["status"] == "Ready":
            st.success(
                f"{report['name']} • Ready • "
                f"{report['readable_pages']}/"
                f"{report['total_pages']} readable page(s)"
            )
        else:
            st.error(
                f"{report['name']} • Not processed"
            )

        st.markdown(
            f"**Reason:** {report['reason']}"
        )

        if report["recommendation"]:
            st.caption(
                "Recommended action: "
                + report["recommendation"]
            )

        if report_index < len(processing_reports):
            st.divider()


if not all_documents:
    st.error(
        "No readable document content is available. "
        "Review the document health report."
    )
    st.stop()


# =========================================================
# 18. RAG INDEX
# =========================================================

document_chunks = create_chunks(
    all_documents
)

with st.spinner(
    "Building semantic document index..."
):
    vector_index = create_vector_index(
        document_chunks
    )


document_names = list(
    dict.fromkeys(
        page["document"]
        for page in all_documents
    )
)


with st.expander(
    "Technical details",
    expanded=False,
):
    detail_one, detail_two, detail_three = st.columns(3)

    detail_one.metric(
        "Readable Pages",
        len(all_documents),
    )

    detail_two.metric(
        "RAG Chunks",
        len(document_chunks),
    )

    detail_three.metric(
        "Indexed Chunks",
        vector_index.ntotal
        if vector_index is not None
        else 0,
    )

    preview_tab, chunks_tab = st.tabs(
        [
            "Extracted Text",
            "RAG Chunks",
        ]
    )

    with preview_tab:
        for preview_index, page in enumerate(
            all_documents[:5]
        ):
            st.markdown(
                f"**{page['document']} — "
                f"Page {page['page']}**"
            )

            st.text_area(
                "Extracted text",
                value=page["text"][:2000],
                height=170,
                disabled=True,
                key=(
                    f"preview_{preview_index}_"
                    f"{page['document']}_{page['page']}"
                ),
                label_visibility="collapsed",
            )

    with chunks_tab:
        for chunk_index, chunk in enumerate(
            document_chunks[:5]
        ):
            st.markdown(
                f"**{chunk['document']} — "
                f"Page {chunk['page']} — "
                f"Chunk {chunk['chunk']}**"
            )

            st.text_area(
                "Chunk text",
                value=chunk["text"],
                height=150,
                disabled=True,
                key=(
                    f"chunk_{chunk_index}_"
                    f"{chunk['document']}_"
                    f"{chunk['page']}_"
                    f"{chunk['chunk']}"
                ),
                label_visibility="collapsed",
            )


# =========================================================
# 19. DOCUMENT INTELLIGENCE WORKSPACE
# =========================================================

st.markdown(
    """
    <div class="section-head">
        <div>
            <div class="section-title">
                Document Intelligence Workspace
            </div>
            <div class="section-sub">
                Summarize, compare or query your processed documents.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

summary_tab, compare_tab, ask_tab = st.tabs(
    [
        "▤  Summary",
        "⇄  Compare",
        "⌕  Ask AI",
    ]
)


# =========================================================
# 20. SUMMARY
# =========================================================

with summary_tab:
    st.subheader("AI Document Summary")

    st.caption(
        "Generate a structured professional report "
        "for every readable uploaded document."
    )

    if st.button(
        "Generate Summaries",
        type="primary",
        key="summarize_documents",
    ):
        grouped_documents = {}

        for page in all_documents:
            grouped_documents.setdefault(
                page["document"],
                [],
            ).append(page)

        for document_name, pages in grouped_documents.items():
            with st.spinner(
                f"Analyzing {document_name}..."
            ):
                summary = summarize_document(
                    document_name,
                    pages,
                )

            if summary:
                st.markdown(
                    f"### {document_name}"
                )
                st.markdown(summary)
                st.divider()


# =========================================================
# 21. COMPARE
# =========================================================

with compare_tab:
    st.subheader("Multi-Document Comparison")

    st.caption(
        "Select two or more readable documents. "
        "Only the selected files are compared."
    )

    if len(document_names) >= 2:
        selected_documents = st.multiselect(
            "Documents to compare",
            options=document_names,
            default=document_names[:2],
            key="comparison_document_selector",
            help="Select at least two documents.",
        )

        st.caption(
            f"{len(selected_documents)} selected from "
            f"{len(document_names)} readable document(s)."
        )

        if st.button(
            "Compare Selected Documents",
            type="primary",
            key="compare_documents",
        ):
            if len(selected_documents) < 2:
                st.warning(
                    "Select at least two documents."
                )
            else:
                selected_pages = [
                    page
                    for page in all_documents
                    if page["document"]
                    in selected_documents
                ]

                with st.spinner(
                    f"Comparing {len(selected_documents)} documents..."
                ):
                    comparison = compare_documents(
                        selected_pages
                    )

                if comparison:
                    st.markdown(
                        "### Comparative Analysis"
                    )

                    st.caption(
                        "Compared: "
                        + " • ".join(
                            selected_documents
                        )
                    )

                    st.markdown(comparison)

    else:
        st.info(
            "At least two readable PDFs are required "
            "for comparison."
        )


# =========================================================
# 22. ASK AI
# =========================================================

with ask_tab:
    st.subheader("Ask Your Documents")

    st.caption(
        "DocuMind retrieves relevant evidence first, "
        "then generates a grounded answer."
    )

    question = st.text_input(
        "Question",
        placeholder=(
            "Example: What are the main findings "
            "and limitations?"
        ),
        key="document_question",
    )

    if st.button(
        "Ask DocuMind",
        type="primary",
        key="ask_documind",
    ):
        if not question.strip():
            st.warning(
                "Enter a question first."
            )

        elif vector_index is None:
            st.error(
                "The semantic search index is unavailable."
            )

        else:
            with st.spinner(
                "Searching documents and generating a grounded answer..."
            ):
                relevant_chunks = retrieve_relevant_chunks(
                    question=question,
                    index=vector_index,
                    chunks=document_chunks,
                    top_k=5,
                    similarity_threshold=0.25,
                )

                if not relevant_chunks:
                    st.warning(
                        "No sufficiently relevant evidence "
                        "was found in the uploaded documents."
                    )
                else:
                    context = "\n\n".join(
                        (
                            f"DOCUMENT: {chunk['document']}\n"
                            f"PAGE: {chunk['page']}\n"
                            f"CHUNK: {chunk['chunk']}\n"
                            f"CONTENT:\n{chunk['text']}"
                        )
                        for chunk in relevant_chunks
                    )

                    rag_prompt = f"""
Answer the user's question using ONLY the retrieved
document context below.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Rules:
1. Use only retrieved document evidence.
2. Do not use outside knowledge.
3. Never invent facts.
4. If evidence is insufficient, say that the uploaded
   documents do not contain enough information.
5. Give a clear professional answer.
6. Do not invent document names or page numbers.
7. Mention the supporting document/page when useful.
"""

                    answer = ask_llm(
                        rag_prompt
                    )

                    if answer:
                        st.markdown(
                            "### Answer"
                        )

                        st.markdown(answer)

                        st.markdown(
                            "### Sources"
                        )

                        seen_sources = set()

                        for chunk in relevant_chunks:
                            source = (
                                chunk["document"],
                                chunk["page"],
                            )

                            if source in seen_sources:
                                continue

                            seen_sources.add(source)

                            st.markdown(
                                f"**{chunk['document']}** "
                                f"· Page {chunk['page']}"
                            )

                        with st.expander(
                            "Retrieved evidence",
                            expanded=False,
                        ):
                            for evidence_number, chunk in enumerate(
                                relevant_chunks,
                                start=1,
                            ):
                                st.markdown(
                                    f"**Evidence {evidence_number} — "
                                    f"{chunk['document']} — "
                                    f"Page {chunk['page']} — "
                                    f"Chunk {chunk['chunk']}**"
                                )

                                st.caption(
                                    f"Hybrid: "
                                    f"{chunk['hybrid_score']:.3f} • "
                                    f"Semantic: "
                                    f"{chunk['semantic_score']:.3f} • "
                                    f"Exact terms: "
                                    f"{chunk['lexical_score']:.3f} • "
                                    f"Document match: "
                                    f"{chunk['document_score']:.3f}"
                                )

                                st.text_area(
                                    "Evidence",
                                    value=chunk["text"],
                                    height=190,
                                    disabled=True,
                                    key=(
                                        f"evidence_"
                                        f"{evidence_number}_"
                                        f"{chunk['document']}_"
                                        f"{chunk['page']}_"
                                        f"{chunk['chunk']}"
                                    ),
                                    label_visibility="collapsed",
                                )


st.divider()

st.caption(
    "DocuMind AI • Multi-Document Intelligence • "
    "Evidence-grounded document analysis."
)
