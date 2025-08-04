import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from models import DocumentChunk

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", vector_db_path: str = "data/vector_db"):
        print(f"Initializing VectorStore with model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            print("SentenceTransformer model loaded successfully")
        except Exception as e:
            print(f"Error loading SentenceTransformer: {e}")
            raise
        
        self.vector_db_path = vector_db_path
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        os.makedirs(vector_db_path, exist_ok=True)
        
        self.index_path = os.path.join(vector_db_path, "faiss_index.bin")
        self.metadata_path = os.path.join(vector_db_path, "metadata.pkl")
        
        self.load_or_create_index()
    
    def load_or_create_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"Loaded existing index with {len(self.metadata)} chunks")
            except Exception as e:
                print(f"Error loading existing index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        print("Created new FAISS index")
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        if not chunks:
            return
        
        print(f"Adding {len(chunks)} chunks to vector store")
        texts = [chunk.content for chunk in chunks]
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            faiss.normalize_L2(embeddings)
            
            self.index.add(embeddings.astype(np.float32))
            
            for chunk in chunks:
                self.metadata.append({
                    'content': chunk.content,
                    'filename': chunk.filename,
                    'page_number': chunk.page_number,
                    'chunk_id': chunk.chunk_id
                })
            
            self.save_index()
            print(f"Successfully added {len(chunks)} chunks")
            
        except Exception as e:
            print(f"Error adding chunks: {e}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        if len(self.metadata) == 0:
            print("No documents in vector store")
            return []
        
        try:
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding.astype(np.float32), min(k, len(self.metadata)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx < len(self.metadata):
                    results.append((self.metadata[idx], float(score)))
            
            print(f"Found {len(results)} relevant chunks for query")
            return results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def save_index(self):
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            print(f"Error saving index: {e}")
