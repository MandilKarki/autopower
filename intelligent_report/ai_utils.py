# AI and RAG Utilities
import os
import re
import ollama
from typing import List, Dict, Any, Optional, Union
import chromadb


class AIManager:
    """
    Manages interactions with AI models and embeddings
    """
    
    def __init__(self, model_name: str = None):
        self.available_models = self._get_available_models()
        
        # Set default model if none provided or if provided model isn't available
        if model_name is None or model_name not in self.available_models:
            self.model_name = self.available_models[0] if self.available_models else "llama2"
            if model_name is not None and model_name not in self.available_models:
                print(f"Warning: Model {model_name} not available. Using {self.model_name} instead.")
        else:
            self.model_name = model_name
            
        print(f"Using AI model: {self.model_name}")
    
    def _get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            response = ollama.list()
            return [model['name'] for model in response['models']]
        except Exception as e:
            print(f"Warning: Could not retrieve model list from Ollama: {e}")
            return []
    
    def generate_content(self, prompt: str) -> str:
        """Generate content using the AI model"""
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ])
            
            return response['message']['content']
        except Exception as e:
            print(f"Error generating content: {e}")
            return f"Error: {str(e)}"
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for text using the AI model"""
        try:
            embedding_response = ollama.embeddings(
                model=self.model_name,
                prompt=text
            )
            
            return embedding_response['embedding']
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return []
    
    def parse_json_response(self, json_text: str) -> Any:
        """Parse JSON from AI response, handling markdown code blocks"""
        import json
        
        # Try to extract JSON if it's embedded in markdown code blocks
        json_matches = re.findall(r'```(?:json)?\s*(.+?)\s*```', json_text, re.DOTALL)
        if json_matches:
            json_text = json_matches[0]
            
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse JSON response: {e}")
            return {"error": "Could not parse JSON", "raw_response": json_text}


class RAGProcessor:
    """
    Handles Retrieval Augmented Generation
    """
    
    def __init__(self, ai_manager: AIManager):
        self.ai_manager = ai_manager
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client()
        try:
            self.collection = self.chroma_client.create_collection("rag_documents")
        except:
            # Collection might already exist
            self.collection = self.chroma_client.get_collection("rag_documents")
    
    def add_document(self, document_id: str, text: str, metadata: Dict = None):
        """Add a document to the RAG collection"""
        embedding = self.ai_manager.get_embeddings(text)
        
        if not embedding:
            print(f"Warning: Could not get embeddings for document {document_id}")
            return False
        
        try:
            self.collection.add(
                ids=[document_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata or {}]
            )
            return True
        except Exception as e:
            print(f"Error adding document to RAG: {e}")
            return False
    
    def query(self, query_text: str, n_results: int = 3) -> Dict:
        """Query the RAG collection"""
        query_embedding = self.ai_manager.get_embeddings(query_text)
        
        if not query_embedding:
            return {"error": "Could not get embeddings for query"}
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            return results
        except Exception as e:
            print(f"Error querying RAG: {e}")
            return {"error": str(e)}
    
    def generate_augmented_response(self, query: str, context_docs) -> str:
        """Generate a response based on retrieved documents"""
        if not context_docs or not context_docs[0]:
            return f"No relevant information found for: {query}"
        
        # Format retrieved information
        context = "\n\n".join(context_docs[0])
        
        prompt = f"""Based on the following information:

{context}

Answer the query: {query}

Provide a well-structured response that synthesizes 
the information in a way that directly addresses the query."""
        
        return self.ai_manager.generate_content(prompt)
