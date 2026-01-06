import streamlit as st
import tempfile
import os
import re

st.set_page_config(page_title="📄 Document Q&A", layout="wide")
st.title("📄 Document Question Answering")
st.write("Upload a TXT file and ask questions about it.")

uploaded_file = st.file_uploader("Upload TXT file", type=["txt"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(uploaded_file.getbuffer())
        file_path = f.name

    if st.button("📥 Process File"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            st.session_state.text = text
            st.session_state.ready = True
            st.success("✅ File processed successfully")

            os.unlink(file_path)

        except Exception as e:
            st.error(e)

if st.session_state.get("ready", False):
    st.divider()
    st.header("❓ Ask a Question")

    question = st.text_input("Enter your question")

    if st.button("🔍 Get Answer") and question:
        sentences = re.split(r'[.!?]', st.session_state.text)
        keywords = [w.lower() for w in question.split() if len(w) > 3]

        matches = []
        for s in sentences:
            if any(k in s.lower() for k in keywords):
                matches.append(s.strip())

        st.subheader("📌 Answer")
        if matches:
            for i, m in enumerate(matches[:3]):
                st.write(f"{i+1}. {m}")
        else:
            st.write("No direct answer found in the document.")

