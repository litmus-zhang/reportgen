# app.py
import os
import tempfile
from pathlib import Path
import streamlit as st
from unstructured.partition.auto import partition
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
import numpy as np

# ---------- Helpers ----------
def parse_document(file_obj):
    """
    Use unstructured to partition the uploaded file (file-like).
    Returns (full_text, list_of_chunks, table_html_list).
    """
    # Some Streamlit UploadedFile objects work directly with unstructured.partition.
    # If not, write to a temp file and pass filename=...
    try:
        elements = partition(file=file_obj)
    except Exception:
        # fallback: write to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(file_obj.read())
        tmp.flush()
        tmp.close()
        elements = partition(filename=tmp.name)
        os.unlink(tmp.name)

    texts = []
    table_htmls = []
    for el in elements:
        try:
            t = str(el)
        except Exception:
            t = ""
        if t and t.strip():
            texts.append(t.strip())

        md = getattr(el, "metadata", None)
        text_as_html = None
        if md:
            text_as_html = getattr(md, "text_as_html", None) or (
                md.get("text_as_html") if isinstance(md, dict) else None
            )
        if text_as_html:
            table_htmls.append(text_as_html)

    full_text = "\n\n".join(texts)
    chunks = chunk_text(full_text, chunk_size=2000, overlap=200)
    return full_text, chunks, table_htmls


def chunk_text(text, chunk_size=2000, overlap=200):
    if not text:
        return []
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        chunks.append(text[i : i + chunk_size])
        i += chunk_size - overlap
    return chunks


def embed_texts_with_google(texts, embeddings_client):
    """
    Use GoogleGenerativeAIEmbeddings to embed a list of texts.
    """
    if not texts:
        return []
    return embeddings_client.embed_documents(texts)


def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def retrieve_top_k_with_google(query, chunks, chunk_embeddings, embeddings_client, k=3):
    """
    Use embeddings_client.embed_query(...) to get query embedding, then compute cosine sim.
    Returns (top_chunks, scores)
    """
    if not chunks or not chunk_embeddings:
        return [], []

    q_emb = embeddings_client.embed_query(query)
    arr = np.array(chunk_embeddings)  # shape (n_chunks, dim)
    q = np.array(q_emb)
    sims = (arr @ q) / (np.linalg.norm(arr, axis=1) * (np.linalg.norm(q) + 1e-10))
    idx = np.argsort(sims)[-k:][::-1]
    return [chunks[i] for i in idx], [float(sims[i]) for i in idx]


def call_google_llm_with_context(llm, system_prompt, user_prompt, context_text):
    """
    Use ChatGoogleGenerativeAI.invoke(messages). Returns the model text.
    """
    if llm is None:
        return "LLM not configured. Provide your Google API key in the sidebar."

    # Put context into the human message (or system). Keep system for format/instructions.
    human_body = f"CONTEXT:\n{context_text}\n\nUSER:\n{user_prompt}"
    messages = [("system", system_prompt), ("human", human_body)]
    resp = llm.invoke(messages)
    # LangChain's ChatGoogleGenerativeAI returns an AIMessage with .content attribute
    if hasattr(resp, "content"):
        return resp.content
    # fallback to str
    return str(resp)


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Doc → Chat Report Generator (Google Gemini)", layout="wide")
st.title("📄 → 💬 Report generator (Streamlit + Google Generative AI)")

# --- sidebar: Google API key ---
google_api_key = st.sidebar.text_input("Google API key", type="password", help="You can set GOOGLE_API_KEY env var instead")
model_choice = st.sidebar.selectbox("LLM model", options=["models/gemini-2.5-flash", "models/gemini-1.5-pro"], index=0)
embedding_model_choice = st.sidebar.text_input("Embedding model", value="models/gemini-embedding-001")

# initialize or reuse LLM and embeddings in session_state
if "google_llm" not in st.session_state or st.session_state.get("google_api_key") != google_api_key:
    st.session_state.google_llm = None
    st.session_state.google_embeddings = None
    st.session_state.google_api_key = google_api_key

if google_api_key:
    try:
        if st.session_state.google_llm is None:
            st.session_state.google_llm = ChatGoogleGenerativeAI(
                model=model_choice, temperature=0.1, google_api_key=google_api_key
            )
        if st.session_state.google_embeddings is None:
            st.session_state.google_embeddings = GoogleGenerativeAIEmbeddings(
                model=embedding_model_choice, google_api_key=google_api_key
            )
    except Exception as e:
        st.error(f"Failed to create Google clients: {e}")

# session_state for app data
st.session_state.setdefault("messages", [])
st.session_state.setdefault("doc_text", "")
st.session_state.setdefault("doc_chunks", [])
st.session_state.setdefault("chunk_embeddings", [])

# Sidebar file uploader
with st.sidebar:
    st.header("Upload & parse")
    uploaded = st.file_uploader("Upload a document (pdf, docx, txt, pptx...)", type=None)
    if uploaded is not None:
        st.info("Parsing document (may take a few seconds)...")
        full_text, chunks, table_htmls = parse_document(uploaded)
        st.session_state.doc_text = full_text
        st.session_state.doc_chunks = chunks

        if chunks and st.session_state.google_embeddings is not None:
            with st.spinner("Creating embeddings for chunks..."):
                # embed via Google embeddings
                embeddings = embed_texts_with_google(chunks, st.session_state.google_embeddings)
                st.session_state.chunk_embeddings = embeddings
        st.success(
            "Parsed: {} chars, {} chunks, {} tables".format(len(full_text), len(chunks), len(table_htmls))
        )
        if table_htmls:
            st.markdown("**Extracted table previews:**")
            for html in table_htmls[:3]:
                st.write(html, unsafe_allow_html=True)

st.markdown("---")

# Chat render
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_input = st.chat_input("Ask about the document or type 'Generate executive report'...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # choose RAG vs direct
    if st.session_state.doc_chunks and st.session_state.chunk_embeddings and st.session_state.google_embeddings:
        retrieved, scores = retrieve_top_k_with_google(
            user_input,
            st.session_state.doc_chunks,
            st.session_state.chunk_embeddings,
            st.session_state.google_embeddings,
            k=3,
        )
        context = "\n\n---\n\n".join(retrieved)
    else:
        context = st.session_state.doc_text or "No document uploaded; only use user message."

    system_prompt = (
        "You are a helpful technical report writer. Produce a clear, structured report with sections: "
        "Executive Summary, Key Findings, Supporting Evidence, Tables (if any), and Recommendations. "
        "If the user asked a focused question, answer it concisely and cite which part of the document you used."
    )

    with st.chat_message("assistant"):
        st.write("Generating…")

    assistant_text = call_google_llm_with_context(st.session_state.google_llm, system_prompt, user_input, context)
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
    st.experimental_rerun()
