import groq
import os
from typing import List, Dict, Tuple
from models import ChatResponse

class ChatHandler:
    def __init__(self, groq_api_key: str, vector_store):
        self.client = groq.Groq(api_key=groq_api_key)
        self.vector_store = vector_store
    
    def generate_response(self, question: str, top_k: int = 5) -> ChatResponse:
        try:
            # Retrieve relevant chunks
            results = self.vector_store.search(question, k=top_k)
            
            if not results:
                return ChatResponse(
                    answer="I couldn't find relevant information in the uploaded documents. Please make sure you have uploaded PDF documents first.",
                    sources=[]
                )
            
            # Prepare context and sources
            context_parts = []
            sources = []
            
            for chunk_data, score in results:
                context_parts.append(f"From {chunk_data['filename']} (Page {chunk_data['page_number']}):\n{chunk_data['content']}")
                
                sources.append({
                    'filename': chunk_data['filename'],
                    'page_number': chunk_data['page_number'],
                    'snippet': chunk_data['content'][:200] + "..." if len(chunk_data['content']) > 200 else chunk_data['content'],
                    'relevance_score': round(float(score), 3)
                })
            
            context = "\n\n".join(context_parts)
            
            # Create prompt
            prompt = f"""Based on the following context from uploaded PDF documents, answer the user's question accurately and comprehensively.

Context:
{context}

Question: {question}

Instructions:
- Answer based solely on the provided context
- If the context doesn't contain enough information, say so clearly
- Be specific and cite the source documents
- Keep your answer concise but complete

Answer:"""
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model="gemma2-9b-it",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
                top_p=1,
                stream=False
            )
            
            answer = response.choices[0].message.content
            
            return ChatResponse(answer=answer, sources=sources)
            
        except Exception as e:
            print(f"Error in generate_response: {e}")
            return ChatResponse(
                answer=f"I encountered an error while processing your question: {str(e)}. Please try again.",
                sources=[]
            )
