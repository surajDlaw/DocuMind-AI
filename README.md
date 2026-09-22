# DocuMind AI

**AI-powered multi-document analysis using Retrieval-Augmented Generation (RAG).**

DocuMind AI allows users to upload multiple PDF documents, generate summaries, compare documents, and ask questions based on retrieved document evidence.

## Features

- Upload and analyze multiple PDFs
- AI-generated document summaries
- Compare user-selected documents
- Ask questions across uploaded documents
- Document-aware retrieval
- Semantic + keyword-based search
- Source and page-level evidence
- PDF validation and document health reporting

## Tech Stack

- **Python**
- **Streamlit**
- **Groq API**
- **model="openai/gpt-oss-120b"**
- **Sentence Transformers / MiniLM**
- **FAISS**
- **PyMuPDF**

## RAG Pipeline

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
MiniLM Embeddings
    ↓
FAISS Vector Index
    ↓
Document-Aware Retrieval
    ↓
Relevant Evidence
    ↓
Llama 3.3 70B
    ↓
Grounded Answer + Sources
```

## Run Locally

Clone the repository:

```bash
git clone https://github.com/surajDlaw/DocuMind-AI.git
cd DocuMind-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Run:

```bash
streamlit run app.py
```



B.Tech Computer Science & Engineering (Artificial Intelligence)

GitHub: `surajDlaw`
