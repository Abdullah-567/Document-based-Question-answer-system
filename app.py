import streamlit as st
import tempfile
import os
from pypdf import PdfReader
from transformers import pipeline

# ---------------- UI ----------------
st.set_page_config(page_title="📄 Document Q&A", layout="wide")
st.title("📄 Document Question Answering")
st.write("Upload a PDF or TXT file and ask questions.")

# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad"
    )

qa_model = load_model()

# ---------------- Upload ----------------
uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(uploaded_file.getbuffer())
        file_path = f.name

    if st.button("📥 Process File"):
        try:
            if uploaded_file.name.endswith(".pdf"):
                reader = PdfReader(file_path)
                text = " ".join(page.extract_text() or "" for page in reader.pages)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

            st.session_state.document_text = text
            st.session_state.ready = True
            st.success("✅ Document processed successfully")

            os.unlink(file_path)

        except Exception as e:
            st.error(e)

# ---------------- Q&A ----------------
if st.session_state.get("ready", False):
    st.divider()
    st.header("❓ Ask a Question")

    question = st.text_input("Enter your question")

    if st.button("🤖 Get Answer") and question:
        with st.spinner("Answering..."):
            result = qa_model(
                question=question,
                context=st.session_state.document_text[:4000]
            )

            st.subheader("📌 Answer")
            st.write(result["answer"])



