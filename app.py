import streamlit as st
import tempfile
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import pipeline
from pypdf import PdfReader

# ------------------ UI ------------------
st.set_page_config(page_title="📄 Document Q&A", layout="wide")
st.title("📄 Document Q&A App")
st.write("Upload a PDF or TXT file, then ask questions about it.")

# ------------------ Upload ------------------
uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as f:
        f.write(uploaded_file.getbuffer())
        file_path = f.name

    if st.button("📥 Process File"):
        with st.spinner("Processing document..."):
            try:
                # Read file
                if file_path.endswith(".pdf"):
                    reader = PdfReader(file_path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()

                # Split text
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=100
                )
                chunks = splitter.split_text(text)

                # Embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                # Vector store
                vectordb = FAISS.from_texts(chunks, embeddings)

                # Save to session
                st.session_state.vectordb = vectordb
                st.session_state.file_ready = True

                st.success(f"✅ File processed successfully! ({len(chunks)} chunks)")
                os.unlink(file_path)

            except Exception as e:
                st.error(f"Error processing file: {e}")

# ------------------ Q&A ------------------
if st.session_state.get("file_ready", False):
    st.divider()
    st.header("❓ Ask a Question")

    question = st.text_input("Type your question here")

    if st.button("🤖 Get Answer") and question:
        with st.spinner("Thinking..."):
            try:
                # Load local model
                qa_pipeline = pipeline(
                    "text2text-generation",
                    model="google/flan-t5-base",
                    max_length=256
                )

                llm = HuggingFacePipeline(pipeline=qa_pipeline)

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=st.session_state.vectordb.as_retriever(),
                    chain_type="stuff"
                )

                result = qa_chain.run(question)

                st.subheader("📌 Answer")
                st.write(result)

            except Exception as e:
                st.error(f"Error answering question: {e}")
else:
    st.info("👆 Upload and process a document to start asking questions.")

