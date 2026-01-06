
import streamlit as st
import tempfile
import os
import sys
import subprocess
import time

# Page configuration
st.set_page_config(
    page_title="Document QA System",
    page_icon="📚",
    layout="wide"
)

# Title
st.title("📚 Document Question Answering System")
st.write("Upload PDF/TXT files and ask questions")

# Initialize session state
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk Size", 200, 1000, 500)
    
    if st.session_state.processed:
        st.success("✅ Ready")
    else:
        st.info("📁 Upload files")

# Main interface
st.header("📁 Upload Documents")
uploaded_files = st.file_uploader(
    "Choose PDF/TXT files",
    type=['pdf', 'txt'],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📄 {len(uploaded_files)} files selected")
    
    if st.button("🚀 Process Documents", type="primary"):
        with st.spinner("Processing documents..."):
            try:
                # Save uploaded files
                temp_paths = []
                for file in uploaded_files:
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False, 
                        suffix=os.path.splitext(file.name)[1]
                    )
                    temp_file.write(file.getbuffer())
                    temp_file.close()
                    temp_paths.append(temp_file.name)
                
                # Import required libraries
                try:
                    from langchain_community.document_loaders import PyPDFLoader, TextLoader
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    from langchain_community.vectorstores import Chroma
                    from langchain.llms import HuggingFacePipeline
                    from langchain.chains import RetrievalQA
                    from transformers import pipeline
                except ImportError as e:
                    st.error(f"❌ Missing package: {e}")
                    st.info("""
                    Please ensure your requirements.txt contains:
                    - sentence-transformers
                    - transformers
                    - langchain
                    - langchain-community
                    - chromadb
                    - pypdf
                    """)
                    st.stop()
                
                # Load and process documents
                documents = []
                for file_path in temp_paths:
                    if file_path.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                    elif file_path.endswith('.txt'):
                        loader = TextLoader(file_path)
                    else:
                        continue
                    
                    docs = loader.load()
                    documents.extend(docs)
                
                # Split documents
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=50
                )
                chunks = splitter.split_documents(documents)
                
                # Create embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                
             
                # Replace Chroma with FAISS
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)
                # Load LLM
                llm_pipeline = pipeline(
                    "text2text-generation",
                    model="google/flan-t5-small",
                    max_length=200
                )
                
                llm = HuggingFacePipeline(pipeline=llm_pipeline)
                
                # Create QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=vectorstore.as_retriever()
                )
                
                st.session_state.qa_system = qa_chain
                st.session_state.processed = True
                
                st.success(f"✅ Processed {len(chunks)} chunks!")
                
                # Clean up
                for path in temp_paths:
                    try:
                        os.unlink(path)
                    except:
                        pass
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Question section
if st.session_state.processed:
    st.header("💬 Ask Questions")
    
    # Sample questions
    st.markdown("**Try asking:**")
    cols = st.columns(2)
    sample_questions = [
        "What is the main topic?",
        "Can you summarize this?",
        "What are the key points?",
        "Who is the author?"
    ]
    
    for idx, question in enumerate(sample_questions):
        with cols[idx % 2]:
            if st.button(question, key=f"sample_{idx}"):
                st.session_state.auto_question = question
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        value=getattr(st.session_state, 'auto_question', ''),
        placeholder="e.g., What is this document about?"
    )
    
    if st.button("🔍 Get Answer", type="primary") and question:
        with st.spinner("Searching..."):
            try:
                answer = st.session_state.qa_system.run(question)
                st.markdown("### 📋 Answer")
                st.info(answer)
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.warning("👈 Please upload and process documents first")

st.markdown("---")
st.markdown("Built with LangChain & Streamlit")
