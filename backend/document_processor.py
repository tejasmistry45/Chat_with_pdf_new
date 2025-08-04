import PyPDF2
import os
import uuid
from typing import List, Dict
from models import DocumentChunk

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, file_path: str) -> List[Dict]:
        """Extract text from PDF with better error handling"""
        print(f"Extracting text from: {file_path}")
        pages_text = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                print(f"PDF has {num_pages} pages")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            pages_text.append({
                                'text': text.strip(),
                                'page_number': page_num
                            })
                            print(f"Page {page_num}: {len(text)} characters extracted")
                        else:
                            print(f"Page {page_num}: No text content")
                    except Exception as e:
                        print(f"Error extracting text from page {page_num}: {e}")
                        continue
        
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            raise Exception(f"Failed to read PDF: {str(e)}")
        
        if not pages_text:
            raise Exception("No text content could be extracted from any page")
        
        print(f"Successfully extracted text from {len(pages_text)} pages")
        return pages_text
    
    def create_chunks(self, pages_text: List[Dict], filename: str) -> List[DocumentChunk]:
        """Split text into chunks with better handling"""
        chunks = []
        
        for page_data in pages_text:
            text = page_data['text']
            page_num = page_data['page_number']
            
            # Split into chunks
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                
                # Try to break at sentence boundary
                if end < len(text):
                    for i in range(end, max(start, end - 100), -1):
                        if text[i] in '.!?\n':
                            end = i + 1
                            break
                
                chunk_text = text[start:end].strip()
                if chunk_text and len(chunk_text) > 50:  # Only keep substantial chunks
                    chunk = DocumentChunk(
                        content=chunk_text,
                        filename=filename,
                        page_number=page_num,
                        chunk_id=str(uuid.uuid4())
                    )
                    chunks.append(chunk)
                
                start = end - self.chunk_overlap if end < len(text) else end
        
        print(f"Created {len(chunks)} chunks from {filename}")
        return chunks
    
    def process_pdf(self, file_path: str, filename: str) -> List[DocumentChunk]:
        """Complete PDF processing pipeline with validation"""
        if not os.path.exists(file_path):
            raise Exception(f"File does not exist: {file_path}")
        
        if os.path.getsize(file_path) == 0:
            raise Exception(f"File is empty: {file_path}")
        
        pages_text = self.extract_text_from_pdf(file_path)
        chunks = self.create_chunks(pages_text, filename)
        
        if not chunks:
            raise Exception("No valid chunks could be created from the PDF")
        
        return chunks
