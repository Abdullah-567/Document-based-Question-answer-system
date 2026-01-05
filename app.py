import streamlit as st
import tempfile
import os

st.set_page_config(page_title="Document QA", layout="wide")
st.title("📚 Document Question Answering System")
st.write("Upload PDF/TXT files and ask questions")

# Initialize
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Sidebar
with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk Size", 200, 1000, 500)
    
    if st.session_state.processed:
        st.success("✅ Ready")
    else:
        st.info("📁 Upload files")

# Upload section
st.header("📁 Upload Documents")
uploaded_files = st.file_uploader(
    "Choose PDF/TXT files",
    type=['pdf', 'txt'],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"{len(uploaded_files)} files selected")
    
    if st.button("🚀 Process Documents", type="primary"):
        with st.spinner("Processing..."):
            try:
                # Save files
                temp_paths = []
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, 
                                                   suffix=os.path.splitext(file.name)[1]) as f:
                        f.write(file.getbuffer())
                        temp_paths.append(f.name)
                
                # Import and process
                from langchain_community.document_loaders import PyPDFLoader, TextLoader
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                from langchain_community.embeddings import HuggingFaceEmbeddings
                from langchain_community.vectorstores import Chroma
                from langchain.llms import HuggingFacePipeline
                from langchain.chains import RetrievalQA
                from transformers import pipeline
                
                # Load documents
                documents = []
                for path in temp_paths:
                    if path.endswith('.pdf'):
                        docs = PyPDFLoader(path).load()
                    elif path.endswith('.txt'):
                        docs = TextLoader(path).load()
                    else:
                        continue
                    documents.extend(docs)
                
                # Split
                splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
                chunks = splitter.split_documents(documents)
                
                # Embeddings
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = Chroma.from_documents(chunks, embeddings)
                
                # LLM
                llm = HuggingFacePipeline(pipeline=pipeline(
                    "text2text-generation",
                    model="google/flan-t5-small",
                    max_length=200
                ))
                
                # QA Chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=vectorstore.as_retriever()
                )
                
                st.session_state.qa_chain = qa_chain
                st.session_state.processed = True
                st.success(f"✅ Processed {len(chunks)} chunks!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Question section
if st.session_state.processed:
    st.header("💬 Ask Questions")
    question = st.text_input("Enter your question:")
    
    if st.button("🔍 Get Answer") and question:
        with st.spinner("Searching..."):
            try:
                answer = st.session_state.qa_chain.run(question)
                st.info("**Answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.warning("👈 Please upload and process documents first")

st.markdown("---")
st.markdown("Built with LangChain & Streamlit")
