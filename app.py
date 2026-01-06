import streamlit as st
import tempfile
import os
import time

# Page configuration
st.set_page_config(
    page_title="Document QA System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .stButton > button:hover {
        background-color: #2563EB;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #10B981;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #DBEAFE;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #F3F4F6;
        padding: 2rem;
        border-radius: 0.5rem;
        border: 2px solid #D1D5DB;
        margin: 1.5rem 0;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .question-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #F59E0B;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📚 Document Question Answering System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload PDF/TXT files and ask questions about their content</p>', unsafe_allow_html=True)

# Initialize session state
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Sidebar for settings
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    st.markdown("### Document Processing")
    chunk_size = st.slider("Chunk Size (characters)", 200, 1000, 500, 50,
                          help="Smaller chunks = more precise, larger chunks = more context")
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10)
    
    st.markdown("### Search Settings")
    search_k = st.slider("Documents to retrieve", 1, 10, 3,
                        help="How many document chunks to use for answering")
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    
    if st.session_state.documents_processed:
        st.success("✅ Documents Processed")
        st.write(f"📁 Files: {len(st.session_state.uploaded_files)}")
    else:
        st.info("⏳ Waiting for documents")
    
    st.markdown("---")
    st.markdown("### 🆘 Help")
    
    with st.expander("How to use"):
        st.markdown("""
        1. **Upload** PDF or TXT files
        2. **Click** 'Process Documents'
        3. **Wait** for processing to complete
        4. **Ask questions** about your documents
        """)
    
    with st.expander("Sample questions"):
        st.markdown("""
        - What is the main topic?
        - Can you summarize this?
        - What are the key points?
        - Who is the author?
        - What are the conclusions?
        """)
    
    if st.button("🔄 Clear All", type="secondary"):
        st.session_state.qa_system = None
        st.session_state.documents_processed = False
        st.session_state.uploaded_files = []
        st.session_state.conversation_history = []
        st.rerun()

# Main content area - TWO COLUMNS
col1, col2 = st.columns([1, 1])

# COLUMN 1: File Upload and Processing
with col1:
    st.markdown("### 📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose your documents",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF or text files"
    )
    
    if uploaded_files:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"**Files ready to process:** {len(uploaded_files)}")
        
        file_list = ""
        for file in uploaded_files:
            file_size = len(file.getvalue()) / 1024  # Size in KB
            file_list += f"📄 {file.name} ({file_size:.1f} KB)\n"
        
        st.text(file_list)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Process Documents", type="primary", key="process_btn"):
            with st.spinner("Processing documents... This may take a minute."):
                try:
                    # Save uploaded files temporarily
                    temp_files = []
                    for uploaded_file in uploaded_files:
                        # Create temp file
                        temp_file = tempfile.NamedTemporaryFile(
                            delete=False, 
                            suffix=os.path.splitext(uploaded_file.name)[1]
                        )
                        temp_file.write(uploaded_file.getvalue())
                        temp_file.close()
                        temp_files.append(temp_file.name)
                    
                    # Store in session state
                    st.session_state.uploaded_files = temp_files
                    
                    # Import required libraries
                    from langchain_community.document_loaders import PyPDFLoader, TextLoader
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    from langchain_community.vectorstores import Chroma
                    from langchain.llms import HuggingFacePipeline
                    from langchain.chains import RetrievalQA
                    from transformers import pipeline
                    
                    # Load and process documents
                    documents = []
                    for file_path in temp_files:
                        if file_path.endswith('.pdf'):
                            loader = PyPDFLoader(file_path)
                        elif file_path.endswith('.txt'):
                            loader = TextLoader(file_path, encoding='utf-8')
                        else:
                            continue
                        
                        docs = loader.load()
                        documents.extend(docs)
                    
                    # Split documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=len,
                        separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]
                    )
                    
                    splits = text_splitter.split_documents(documents)
                    
                    # Create embeddings
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    
                    # Create vector store
                    vectorstore = Chroma.from_documents(
                        documents=splits,
                        embedding=embeddings
                    )
                    
                    # Load LLM
                    llm_pipeline = pipeline(
                        "text2text-generation",
                        model="google/flan-t5-small",
                        max_length=200,
                        temperature=0.1
                    )
                    
                    llm = HuggingFacePipeline(pipeline=llm_pipeline)
                    
                    # Create QA chain
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(search_kwargs={"k": search_k}),
                        return_source_documents=True
                    )
                    
                    # Store in session state
                    st.session_state.qa_system = qa_chain
                    st.session_state.documents_processed = True
                    
                    # Show success message
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success(f"✅ Successfully processed {len(splits)} document chunks!")
                    st.markdown("You can now ask questions about your documents in the right column.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show sample questions
                    st.markdown("### 💡 Try these questions:")
                    sample_questions = [
                        "What is the main topic?",
                        "Can you summarize the document?",
                        "What are the key points?",
                        "Who is the author (if mentioned)?"
                    ]
                    
                    for i, q in enumerate(sample_questions):
                        st.markdown(f'<div class="question-box">{i+1}. {q}</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
                    st.info("Try with a smaller file or different format.")

# COLUMN 2: Question & Answer Section
with col2:
    st.markdown("### 💬 Ask Questions")
    
    if not st.session_state.documents_processed:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.warning("Please upload and process documents first.")
        st.markdown("""
        👈 **Steps to follow:**
        1. Upload PDF/TXT files in the left panel
        2. Click 'Process Documents' button
        3. Wait for processing to complete
        4. Then come back here to ask questions
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show sample document structure
        with st.expander("📋 What happens after processing?"):
            st.markdown("""
            After processing, this section will show:
            - **Question input field** to type your questions
            - **Get Answer button** to get responses
            - **Answer display area** with formatted answers
            - **Conversation history** of your questions
            """)
    else:
        # Display conversation history
        if st.session_state.conversation_history:
            st.markdown("#### 📝 Recent Questions")
            for i, (question, answer) in enumerate(st.session_state.conversation_history[-3:]):  # Show last 3
                with st.expander(f"Q{i+1}: {question[:60]}..." if len(question) > 60 else f"Q{i+1}: {question}"):
                    st.markdown(f"**Question:** {question}")
                    st.markdown(f"**Answer:** {answer}")
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            placeholder="e.g., What is the main topic of the document?",
            height=100,
            key="question_input"
        )
        
        # Button row
        col_btn1, col_btn2 = st.columns([2, 1])
        
        with col_btn1:
            if st.button("🔍 Get Answer", type="primary", key="get_answer"):
                if question and question.strip():
                    with st.spinner("Searching for answer..."):
                        try:
                            result = st.session_state.qa_system({"query": question})
                            answer = result['result']
                            
                            # Store in conversation history
                            st.session_state.conversation_history.append((question, answer))
                            
                            # Display answer in a nice box
                            st.markdown("#### 📋 Answer")
                            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                            st.markdown(answer)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Show sources if available
                            if 'source_documents' in result and result['source_documents']:
                                with st.expander("📄 View Sources"):
                                    for i, doc in enumerate(result['source_documents'][:2]):  # Show first 2
                                        st.markdown(f"**Source {i+1}:**")
                                        st.info(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                        if hasattr(doc, 'metadata') and doc.metadata:
                                            st.caption(f"From: {doc.metadata.get('source', 'Unknown')}")
                            
                            # Show success message
                            st.success("✅ Answer generated successfully!")
                            
                        except Exception as e:
                            st.error(f"❌ Error getting answer: {str(e)}")
                            st.info("Try rephrasing your question or check if the document has relevant content.")
                else:
                    st.warning("⚠️ Please enter a question first.")
        
        with col_btn2:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.conversation_history = []
                st.rerun()
        
        # Quick question buttons
        st.markdown("#### 🎯 Quick Questions")
        quick_questions = [
            "What is this about?",
            "Summarize the document",
            "List the main points",
            "What are the conclusions?"
        ]
        
        cols = st.columns(2)
        for idx, q in enumerate(quick_questions):
            with cols[idx % 2]:
                if st.button(q, key=f"quick_{idx}"):
                    # Auto-fill the question
                    st.session_state.question_input = q
                    st.rerun()

# Bottom section - Status and info
st.markdown("---")
st.markdown("### 📊 System Information")

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    status_icon = "✅" if st.session_state.documents_processed else "⏳"
    status_text = "Ready" if st.session_state.documents_processed else "Processing"
    st.metric("System Status", f"{status_icon} {status_text}")

with col_info2:
    st.metric("Files Uploaded", len(st.session_state.uploaded_files))

with col_info3:
    st.metric("Questions Asked", len(st.session_state.conversation_history))

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>📚 Document QA System | Built with LangChain, HuggingFace & Streamlit</p>
    <p>💡 Upload PDF/TXT files and ask questions about their content</p>
</div>
""", unsafe_allow_html=True)

# Add a small delay to ensure everything loads
time.sleep(0.1)
