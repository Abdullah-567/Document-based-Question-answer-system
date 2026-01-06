import streamlit as st
import tempfile
import os
import sys

st.set_page_config(page_title="Doc QA", layout="centered")
st.title("📄 Simple Document Q&A")
st.write("Upload a PDF/TXT file and ask questions")

# Initialize
if 'qa' not in st.session_state:
    st.session_state.qa = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Step 1: Upload
uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'txt'])

if uploaded_file and st.button("Process File"):
    with st.spinner("Processing..."):
        try:
            # Save file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            
            # Import
            from langchain_community.document_loaders import PyPDFLoader, TextLoader
            from langchain.text_splitter import CharacterTextSplitter
            from langchain_community.embeddings import HuggingFaceEmbeddings
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
            all_text = ""
            for doc in docs:
                all_text += doc.page_content + "\n"
            
            # Simple splitting
            chunks = [all_text[i:i+500] for i in range(0, len(all_text), 500)]
            
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # SIMPLE VECTOR STORE - Use in-memory
            from langchain_community.vectorstores import FAISS
            from langchain.docstore.in_memory import InMemoryDocstore
            from langchain.schema import Document
            
            # Create documents
            documents = [Document(page_content=chunk) for chunk in chunks]
            
            # Create FAISS index
            vectorstore = FAISS.from_documents(
                documents, 
                embeddings
            )
            
            # Create QA chain
            llm = HuggingFacePipeline(pipeline=pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                max_length=200
            ))
            
            st.session_state.qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )
            st.session_state.processed = True
            
            # Cleanup
            os.unlink(tmp_path)
            
            st.success(f"✅ Ready! Processed {len(chunks)} text chunks")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.text(traceback.format_exc())

# Step 2: Ask questions
if st.session_state.processed:
    st.header("Ask Questions")
    
    question = st.text_input("Your question:")
    
    if st.button("Get Answer") and question:
        with st.spinner("Finding answer..."):
            try:
                answer = st.session_state.qa.run(question)
                st.markdown("**Answer:**")
                st.info(answer)
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.info("👆 Upload a file and click 'Process File'")

st.caption("Simple Document QA System")
