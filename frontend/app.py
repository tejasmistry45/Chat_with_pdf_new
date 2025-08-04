import streamlit as st
import requests
import json
from typing import List, Dict
import io

# Configure Streamlit
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide"
)

# Backend API base URL
API_BASE_URL = "http://localhost:8000"

def upload_files(files):
    """Upload files to backend with proper error handling"""
    try:
        # Prepare files for upload
        files_data = []
        for file in files:
            # Reset file pointer to beginning
            file.seek(0)
            file_content = file.read()
            files_data.append(("files", (file.name, io.BytesIO(file_content), "application/pdf")))
        
        print(f"Uploading {len(files_data)} files to {API_BASE_URL}/upload-pdfs/")
        
        # Make the request
        response = requests.post(
            f"{API_BASE_URL}/upload-pdfs/", 
            files=files_data,
            timeout=300  # 5 minute timeout for large files
        )
        
        print(f"Upload response status: {response.status_code}")
        print(f"Upload response: {response.text}")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Upload failed with status {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Make sure the backend server is running on http://localhost:8000")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ Upload timed out. File might be too large.")
        return None
    except Exception as e:
        st.error(f"❌ Upload error: {str(e)}")
        print(f"Upload exception: {e}")
        return None

def send_chat_message(question: str) -> Dict:
    """Send chat message to backend"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/",
            json={"question": question},
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Chat failed with status {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend for chat.")
        return None
    except Exception as e:
        st.error(f"❌ Chat error: {str(e)}")
        return None

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health/", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def display_sources(sources: List[Dict]):
    """Display source citations"""
    if sources:
        st.subheader("📚 Sources")
        for i, source in enumerate(sources, 1):
            with st.expander(f"Source {i}: {source['filename']} - Page {source['page_number']}"):
                st.write(f"**Relevance Score:** {source['relevance_score']:.3f}")
                st.write(f"**Snippet:** {source['snippet']}")

def main():
    st.title("🤖 Chat with PDF")
    st.markdown("Upload your PDF documents and ask questions about their content!")
    
    # Check backend health
    health = check_backend_health()
    if not health:
        st.error("❌ Backend server is not running. Please start the backend server first.")
        st.stop()
    else:
        st.success(f"✅ Backend connected - Components initialized: {health.get('components_initialized', False)}")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📄 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF documents to chat with",
            key="pdf_uploader"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s):")
            for file in uploaded_files:
                st.write(f"📄 {file.name} ({file.size / 1024:.1f} KB)")
        
        if uploaded_files and st.button("📤 Process Documents", type="primary"):
            with st.spinner("Processing documents... This may take a few minutes."):
                result = upload_files(uploaded_files)
                
                if result:
                    st.success("✅ Documents processed successfully!")
                    st.json(result)
                    
                    # Store successful upload in session state
                    st.session_state.documents_uploaded = True
                    st.session_state.upload_result = result
                else:
                    st.error("❌ Failed to process documents. Check the backend logs for details.")
        
        # Show upload status
        if hasattr(st.session_state, 'documents_uploaded') and st.session_state.documents_uploaded:
            st.success("📚 Documents are loaded and ready for questions!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                display_sources(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = send_chat_message(prompt)
                
                if response:
                    st.markdown(response["answer"])
                    
                    # Store assistant message with sources
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", [])
                    })
                    
                    # Display sources
                    if response.get("sources"):
                        display_sources(response["sources"])
                else:
                    error_msg = "Sorry, I encountered an error. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
