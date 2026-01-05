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
    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk Size", 200, 1000, 500)
    
    if st.session_state.processed:
        st.success("✅ Ready")
    else:
        st.info("📁 Upload files")
    
    st.markdown("---")
    st.markdown("### Tips:")
    st.markdown("""
    1. Upload PDF/TXT files
    2. Click Process Documents
    3. Ask questions
    """)

# Upload section
st.header("📁 Upload Documents")
uploaded_files = st.file_uploader(
    "Choose PDF or TXT files",
    type=['pdf', 'txt'],
    accept_multiple_files=True,
    help="Maximum 200MB total"
)

if uploaded_files:
    st.info(f"📄 {len(uploaded_files)} files selected")
    
    if st.button("🚀 Process Documents", type="primary"):
        with st.spinner("Processing documents... This may take a minute."):
            try:
                # Save uploaded files temporarily
                temp_paths = []
                for file in uploaded_files:
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False, 
                        suffix=os.path.splitext(file.name)[1]
                    )
                    temp_file.write(file.getbuffer())
                    temp_file.close()
                    temp_paths.append(temp_file.name)
                
                # ===== IMPORTANT: Force CPU-only to avoid CUDA errors =====
                import sys
                import subprocess
                
                # Now import the required libraries
                from langchain_community.document_loaders import PyPDFLoader, TextLoader
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                from langchain_community.embeddings import HuggingFaceEmbeddings
                from langchain_community.vectorstores import Chroma
                from langchain.llms import HuggingFacePipeline
                from langchain.chains import RetrievalQA
                from transformers import pipeline
                
                # Load and process documents
                documents = []
                for file_path in temp_paths:
                    if file_path.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                        print(f"Loading PDF: {file_path}")
                    elif file_path.endswith('.txt'):
                        loader = TextLoader(file_path, encoding='utf-8')
                        print(f"Loading text: {file_path}")
                    else:
                        continue
                    
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"  Loaded {len(docs)} pages")
                
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=50,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                
                splits = text_splitter.split_documents(documents)
                print(f"Created {len(splits)} chunks")
                
                # Create embeddings - FORCE CPU
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},  # Force CPU
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                # Create vector store
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    persist_directory=None  # Don't persist to save memory
                )
                
                # Load LLM with CPU-only
                llm_pipeline = pipeline(
                    "text2text-generation",
                    model="google/flan-t5-small",
                    max_length=200,
                    temperature=0.1,
                    device=-1  # Force CPU
                )
                
                llm = HuggingFacePipeline(pipeline=llm_pipeline)
                
                # Create QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=False  # Disable for now to save memory
                )
                
                # Store in session state
                st.session_state.qa_chain = qa_chain
                st.session_state.processed = True
                
                # Clean up temp files
                for path in temp_paths:
                    try:
                        os.unlink(path)
                    except:
                        pass
                
                st.success(f"✅ Successfully processed {len(splits)} document chunks!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Question section
if st.session_state.processed:
    st.header("💬 Ask Questions")
    
    # Sample questions
    sample_questions = [
        "What is the main topic?",
        "Can you summarize this?",
        "What are the key points?",
        "Who is the author?"
    ]
    
    # Display sample questions as buttons
    cols = st.columns(2)
    for idx, question in enumerate(sample_questions):
        with cols[idx % 2]:
            if st.button(question, key=f"sample_{idx}"):
                st.session_state.auto_question = question
    
    # Question input
    question = st.text_input(
        "Or enter your own question:",
        value=getattr(st.session_state, 'auto_question', ''),
        placeholder="e.g., What is this document about?"
    )
    
    if st.button("🔍 Get Answer", type="primary") and question:
        with st.spinner("Searching for answer..."):
            try:
                # Clear auto question
                if hasattr(st.session_state, 'auto_question'):
                    del st.session_state.auto_question
                
                # Get answer
                result = st.session_state.qa_chain({"query": question})
                answer = result['result']
                
                # Display answer
                st.markdown("### 📋 Answer")
                st.markdown(f"""<div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #4B9CD3;'>
                {answer}
                </div>""", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error getting answer: {str(e)}")
else:
    st.warning("👈 Please upload and process documents first")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📚 Document QA System | Built with LangChain & Streamlit</p>
    <p>💡 Upload PDF/TXT files and ask questions about their content</p>
</div>
""", unsafe_allow_html=True)

# Clear cache button in sidebar
with st.sidebar:
    if st.button("🔄 Clear Cache"):
        st.session_state.qa_chain = None
        st.session_state.processed = False
        st.rerun()
        st.success("Cache cleared!")
