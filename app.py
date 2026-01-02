import os
import io
from typing import List
from dataclasses import dataclass, field

import streamlit as st
import pdfplumber
import numpy as np
import chromadb
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# CONFIG
MODEL_NAME = "openai/gpt-oss-20b"
EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBED_DIM = 384

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in environment variables")

client = Groq(api_key=GROQ_API_KEY)

# Chroma DB
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="study_docs")

@dataclass
class DocChunk:
    id: str
    text: str
    meta: dict = field(default_factory=dict)
    embedding: np.ndarray = None


# HELPERS
def extract_text_from_pdf_file(uploaded_file) -> str:
    try:
        uploaded_file.seek(0)
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n\n".join(pages)
    except Exception:
        return ""

def chunk_text(text: str, chunk_size:int=400, overlap:int=50) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += (chunk_size - overlap)
    return chunks

@st.cache_resource(show_spinner=False)
def load_embed_model(name=EMBED_MODEL_NAME):
    return SentenceTransformer(name)

def embed_texts(model, texts: List[str]) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def add_to_chroma(chunks: List[DocChunk]):
    for chunk in chunks:
        collection.add(
            ids=[chunk.id],
            documents=[chunk.text],
            metadatas=[chunk.meta],
            embeddings=[chunk.embedding.tolist()]
        )

def search_chroma(query: str, top_k=4):
    q_emb = embed_texts(st.session_state.embed_model, [query])[0]
    results = collection.query(query_embeddings=[q_emb.tolist()], n_results=top_k)
    return results

def call_groq_chat(system: str, user: str, temperature: float=0.2, max_tokens:int=512) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling Groq API: {e}"

def build_prompt_with_context(question: str, contexts: List[str]) -> str:
    header = (
        "You are a strict study assistant. You must answer the question STRICTLY based on the provided context below. "
        "Do not use any outside knowledge, general information, or pre-training data. "
        "If the answer cannot be found in the context, you MUST say exactly: 'I cannot answer this based on the provided documents.'\n\n"
    )
    ctxs = "\n\n".join([f"Context {i+1}:\n{c}" for i,c in enumerate(contexts)])
    prompt = f"{header}{ctxs}\n\nUser question: {question}\n\nAnswer concisely based ONLY on the context above. Do not hallucinate."
    return prompt

def summarize_all_docs():
    all_docs = [c.text for c in st.session_state.doc_chunks]
    if not all_docs:
        return "No documents available to summarize."
    joined_text = "\n\n".join(all_docs)
    summary_prompt = f"Summarize the following text from multiple documents in clear bullet points:\n\n{joined_text}"
    return call_groq_chat(
        system="You are a summarization expert.",
        user=summary_prompt,
        temperature=0.3,
        max_tokens=800
    )


# UI SETUP
st.set_page_config(page_title="📚 RAG Study Assistant", layout="wide")

# App Title
st.markdown(
    """
    <h1 style="text-align:center;">📚 RAG Study Assistant</h1>
    <p style="text-align:center;color:gray;font-size:18px;">
    Upload your notes or textbooks (PDF/TXT), build an AI-powered index, and chat with your documents.
    </p>
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("📂 Upload & Index")
    uploaded_files = st.file_uploader(
        "Select PDFs or TXT files",
        accept_multiple_files=True,
        type=["pdf", "txt"]
    )
    if uploaded_files:
        st.success(f"📂 {len(uploaded_files)} file(s) uploaded successfully!")

    st.markdown("### ⚙️ Processing Parameters")
    chunk_size = st.slider("Chunk size (words)", value=400, min_value=100, max_value=2000, step=50)
    overlap = st.slider("Chunk overlap (words)", value=50, min_value=0, max_value=1000, step=10)

    btn_build = st.button("🚀 Build / Update Index", use_container_width=True)

    st.markdown("---")
    st.markdown("**💡 Tip:** Larger chunk size = better context, smaller size = more precise search.")


# SESSION INIT
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "embed_model" not in st.session_state:
    st.session_state.embed_model = load_embed_model()


# BUILD INDEX
if btn_build:
    if not uploaded_files:
        st.warning("📂 Please upload at least one file first.")
    else:
        progress = st.progress(0)
        all_chunks = []
        total_files = len(uploaded_files)

        for idx, f in enumerate(uploaded_files, start=1):
            if f.name.lower().endswith(".pdf"):
                text = extract_text_from_pdf_file(f)
            else:
                f.seek(0)
                text = f.read().decode("utf-8")

            if not text.strip():
                continue

            chunked_texts = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            embs = embed_texts(st.session_state.embed_model, chunked_texts)

            for i, c in enumerate(chunked_texts):
                dc = DocChunk(
                    id=f"{f.name}_{i}",
                    text=c,
                    meta={"source": f.name, "chunk": i},
                    embedding=embs[i]
                )
                all_chunks.append(dc)

            progress.progress(idx / total_files)

        add_to_chroma(all_chunks)
        st.session_state.doc_chunks.extend(all_chunks)
        st.success(f"✅ Indexed {len(all_chunks)} chunks from {total_files} file(s).")


# CHAT UI
st.markdown("## 💬 Ask Your Documents")
question = st.text_area("Enter your question:", height=120, placeholder="e.g. Summarize chapter 2...")
top_k = st.slider("Number of context chunks to retrieve", 1, 8, 4)

col1, col2 = st.columns(2)
with col1:
    ask_btn = st.button("🔍 Get Answer", use_container_width=True)
with col2:
    sum_btn = st.button("📝 Summarize All Documents", use_container_width=True)

if ask_btn:
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        results = search_chroma(question, top_k)
        contexts = results["documents"][0]
        prompt = build_prompt_with_context(question, contexts)

        with st.spinner("Thinking..."):
            answer = call_groq_chat(
                system="You are a strict assistant that only answers based on the provided text.",
                user=prompt,
                temperature=0.0,
                max_tokens=600
            )

        st.markdown("### 🧠 Answer")
        st.write(answer)

        with st.expander("📜 Retrieved Contexts"):
            for i, ctx in enumerate(contexts, start=1):
                st.markdown(f"**Context {i}:** {ctx}")

if sum_btn:
    with st.spinner("Summarizing all documents..."):
        summary = summarize_all_docs()
    st.markdown("### 📝 Summary of All Documents")
    st.write(summary)


# DEBUG
with st.expander("🛠 Debug Info"):
    st.json(collection.peek())
