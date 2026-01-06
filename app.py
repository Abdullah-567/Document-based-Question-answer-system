import streamlit as st
import tempfile
import os

# Simple page
st.title("📄 Document Q&A")
st.write("Upload a file and ask questions")

# UPLOAD SECTION
uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=['pdf', 'txt'])

if uploaded_file:
    # Save the file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as f:
        f.write(uploaded_file.getbuffer())
        file_path = f.name
    
    # Process button
    if st.button("Process File"):
        with st.spinner("Reading file..."):
            try:
                # Read file content
                if file_path.endswith('.pdf'):
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(file_path)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                else:  # txt file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                
                # Save to session
                st.session_state.document_text = text
                st.session_state.file_processed = True
                
                # Show stats
                st.success(f"✅ File processed! ({len(text)} characters)")
                st.info(f"First 500 chars: {text[:500]}...")
                
                # Clean up
                os.unlink(file_path)
                
            except Exception as e:
                st.error(f"Error: {e}")

# QUESTION SECTION - ONLY SHOWS AFTER FILE IS PROCESSED
if st.session_state.get('file_processed', False):
    st.divider()
    st.header("Ask Questions")
    
    # Question input
    question = st.text_input("Enter your question:")
    
    if st.button("Get Answer") and question:
        with st.spinner("Finding answer..."):
            try:
                # Get the document text
                text = st.session_state.document_text
                
                # SIMPLE ANSWER: Just find relevant sentences
                import re
                
                # Split into sentences
                sentences = re.split(r'[.!?]+', text)
                
                # Find sentences containing question words
                question_words = question.lower().split()
                relevant_sentences = []
                
                for sentence in sentences:
                    if any(word in sentence.lower() for word in question_words if len(word) > 3):
                        relevant_sentences.append(sentence.strip())
                
                # Show answer
                if relevant_sentences:
                    st.subheader("Answer:")
                    for i, sentence in enumerate(relevant_sentences[:3]):  # Show first 3
                        st.write(f"{i+1}. {sentence}")
                else:
                    st.info("No direct answer found in document.")
                    st.write("Document content preview:")
                    st.write(text[:500] + "...")
                    
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("👆 Upload a file and click 'Process File' first")
