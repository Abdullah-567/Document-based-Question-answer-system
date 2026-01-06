import streamlit as st
import tempfile
import os
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- UI ----------------
st.set_page_config(page_title="📄 Document Q&A", layout="wide")
st.title("📄 Document Question Answering")
st.write("Upload a PDF or TXT file and ask questions about it.")

# ---------------- Load models ----------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    qa_model = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256
    )
    return embedder, qa_model

embedder, qa_model = load_models()

# ---------------- Upload ----------------
uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(uploaded_file.getbuffer())
        file_path = f.name

    if st.button("📥 Process File"):
        with st.spinner("Processing document..."):
            try:
                if uploaded_file.name.endswith(".pdf"):
                    reader = PdfReader(file_path)
                    text = " ".join(
                        page.extract_text() or "" for page in reader.pages
                    )
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()

                # Split text
                chunks = [text[i:i+500] for i in range(0, len(text), 400)]

                # Create embeddings
                embeddings = embedder.encode(chunks)

                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.ready = True

                st.success(f"✅ Document processed ({len(chunks)} chunks)")
                os.unlink(file_path)

            except Exception as e:
                st.error(e)

# ---------------- Q&A ----------------
if st.session_state.get("ready", False):
    st.divider()
    st.header("❓ Ask a Question")

    question = st.text_input("Enter your question")

    if st.button("🤖 Get Answer") and question:
        with st.spinner("Finding answer..."):
            q_embedding = embedder.encode([question])
            scores = cosine_similarity(q_embedding, st.session_state.embeddings)[0]

            top_idx = np.argmax(scores)
            context = st.session_state.chunks[top_idx]

            prompt = f"""
            Answer the question based on the context below.

            Context:
            {context}

            Question:
            {question}
            """

            answer = qa_model(prompt)[0]["generated_text"]

            st.subheader("📌 Answer")
            st.write(answer)


