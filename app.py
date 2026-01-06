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

# Function to check and install missing packages
def check_and_install_packages():
    """Check if required packages are installed, install if missing."""
    required_packages = [
        'langchain',
        'langchain_community', 
        'sentence_transformers',
        'transformers',
        'chromadb',
        'pypdf'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        st.warning(f"📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            # Try to install missing packages
            for package in missing_packages:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            st.success("✅ Packages installed successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to install packages: {e}")
            return False
    return True

# Check packages at startup
if not check_and_install_packages():
    st.error("""
    ❌ Required packages are missing. Please ensure your requirements.txt contains:
    
    ```
    streamlit==1.28.0
    langchain==0.1.0
    langchain-community==0.0.20
    pypdf==3.17.0
    chromadb==0.4.22
    sentence-transformers==2.2.2
    transformers==4.36.0
    ```
    
    Then redeploy the app.
    """)
    st.stop()

# Rest of your app.py code continues here...
# [PASTE THE REST OF YOUR app.py CODE HERE]
