import streamlit as st
import tempfile
import os

# Simple page setup
st.set_page_config(page_title="Doc QA", layout="centered")
st.title("📄 Document Q&A")
st.write("Upload PDF/TXT files and ask questions")

# Initialize
if 'qa' not in st.session_state:
    st.session_state.qa = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

# STEP 1: UPLOAD
st.header("1. Upload File")
uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=['pdf', 'txt'])

if uploaded_file and st.button("📁 Process File"):
    with st.spinner("Processing..."):
        try:
            # Save the file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            
            # Import - simple and safe
            from langchain_community.document_loaders import PyPDFLoader, TextLoader
            from langchain.text_splitter import CharacterTextSplitter
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain.chains import RetrievalQA
            from langchain.llms import HuggingFacePipeline
            from transformers import pipeline
            
            # Load document
            if tmp_path.endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path)
            
            docs = loader.load()
            
            # Split text
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(docs)
            
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Create vector store (FAISS is lighter than Chroma)
            vectorstore = FAISS.from_documents(texts, embeddings)
            
            # Create QA chain
            llm = HuggingFacePipeline(pipeline=pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                max_length=200
            ))
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )
            
            # Save to session
            st.session_state.qa = qa_chain
            st.session_state.processed = True
            
            # Clean up
            os.unlink(tmp_path)
            
            st.success(f"✅ Processed {len(texts)} text chunks!")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# STEP 2: ASK QUESTIONS
if st.session_state.processed:
    st.header("2. Ask Questions")
    
    question = st.text_input("Enter your question:")
    
    if st.button("🔍 Get Answer") and question:
        with st.spinner("Finding answer..."):
            try:
                answer = st.session_state.qa.run(question)
                st.markdown("**Answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.info("👆 Upload a file and click 'Process File' first")

# Footer
st.markdown("---")
st.caption("Simple Document Q&A System")
