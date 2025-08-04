from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class DocumentChunk(BaseModel):
    content: str
    filename: str
    page_number: int
    chunk_id: str

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
