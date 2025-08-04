from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from typing import List
from dotenv import load_dotenv
import sys
import traceback

# Load environment variables
load_dotenv()

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from document_processor import DocumentProcessor
from vector_store import VectorStore
from chat_handler import ChatHandler
from models import ChatRequest, ChatResponse

app = FastAPI(title="PDF Chat API", description="RAG-based PDF Chat Application")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"GROQ API Key configured: {bool(GROQ_API_KEY)}")

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global variables for components
processor = None
vector_store = None
chat_handler = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global processor, vector_store, chat_handler
    try:
        print("Initializing components...")
        processor = DocumentProcessor()
        vector_store = VectorStore()
        if GROQ_API_KEY:
            chat_handler = ChatHandler(GROQ_API_KEY, vector_store)
            print("All components initialized successfully!")
        else:
            print("WARNING: GROQ_API_KEY not found!")
    except Exception as e:
        print(f"Error during startup: {e}")
        traceback.print_exc()

@app.get("/")
async def root():
    return {
        "message": "PDF Chat API is running!",
        "status": "healthy",
        "endpoints": {
            "docs": "/docs",
            "health": "/health/",
            "upload": "/upload-pdfs/",
            "chat": "/chat/"
        }
    }

@app.get("/health/")
async def health_check():
    return {
        "status": "healthy",
        "groq_api_configured": bool(GROQ_API_KEY),
        "components_initialized": all([processor, vector_store, chat_handler])
    }

@app.post("/upload-pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    global processor, vector_store
    
    print(f"Received {len(files)} files for processing")
    
    if not processor or not vector_store:
        raise HTTPException(status_code=500, detail="Components not initialized")
    
    processed_files = []
    
    for file in files:
        print(f"Processing file: {file.filename}")
        
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
        
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        try:
            # Read and save file content
            content = await file.read()
            print(f"File size: {len(content)} bytes")
            
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            
            print(f"File saved to: {file_path}")
            
            # Verify file exists and has content
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                raise Exception("File was not saved properly")
            
            # Process PDF
            print(f"Starting PDF processing for {file.filename}")
            chunks = processor.process_pdf(file_path, file.filename)
            print(f"Created {len(chunks)} chunks")
            
            if not chunks:
                raise Exception("No text content could be extracted from PDF")
            
            # Add to vector store
            print(f"Adding chunks to vector store")
            vector_store.add_chunks(chunks)
            print(f"Successfully added {len(chunks)} chunks to vector store")
            
            processed_files.append({
                'filename': file.filename,
                'chunks_created': len(chunks),
                'file_size': len(content)
            })
            
        except Exception as e:
            print(f"ERROR processing {file.filename}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Clean up failed file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            raise HTTPException(status_code=500, detail=f"Error processing {file.filename}: {str(e)}")
    
    print(f"Successfully processed {len(processed_files)} files")
    return {"message": "Files processed successfully", "files": processed_files}

@app.get("/debug/vector-store-status/")
async def vector_store_status():
    """Debug endpoint to check vector store status"""
    global vector_store
    if not vector_store:
        return {"error": "Vector store not initialized"}
    
    return {
        "total_documents": len(vector_store.metadata),
        "index_size": vector_store.index.ntotal if hasattr(vector_store.index, 'ntotal') else 0,
        "files": list(set([doc['filename'] for doc in vector_store.metadata])) if vector_store.metadata else []
    }

@app.post("/chat/")
async def chat(request: ChatRequest):
    global chat_handler
    
    if not chat_handler:
        raise HTTPException(status_code=500, detail="Chat handler not initialized. Please check GROQ API key.")
    
    try:
        response = chat_handler.generate_response(request.question)
        return response
    except Exception as e:
        print(f"Chat error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
