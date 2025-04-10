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
        # Configure Ollama to use localhost
        ollama.BASE_URL = "http://localhost:11434"
        
        # Get available models
        self.available_models = self._get_available_models()
        
        # Check if llama3.2 is available, otherwise try to pull it
        if "llama3.2" not in self.available_models:
            print("llama3.2 model not found. Attempting to pull it...")
            try:
                # Try to pull llama3 model (this can take time for first run)
                ollama.pull("llama3.2")
                # Refresh model list
                self.available_models = self._get_available_models()
                print("Successfully pulled llama3.2 model")
            except Exception as e:
                print(f"Warning: Could not pull llama3.2 model: {e}")
        
        # Prioritize llama3.2, then user's choice, then available models
        if "llama3.2" in self.available_models:
            self.model_name = "llama3.2"
        elif model_name is not None and model_name in self.available_models:
            self.model_name = model_name
        else:
            # Fallback to first available model or default
            self.model_name = self.available_models[0] if self.available_models else "llama3"
            if model_name is not None and model_name != self.model_name:
                print(f"Warning: Model {model_name} not available. Using {self.model_name} instead.")
        
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
            
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> bool:
        """Add multiple documents to the RAG collection"""
        if not documents:
            return False
            
        # Create document IDs
        doc_ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Get embeddings for all documents
        embeddings = []
        for doc in documents:
            embedding = self.ai_manager.get_embeddings(doc)
            if not embedding:
                print(f"Warning: Could not get embeddings for document")
                continue
            embeddings.append(embedding)
            
        if not embeddings or len(embeddings) != len(documents):
            print(f"Warning: Could not get embeddings for all documents")
            return False
            
        # Ensure metadatas is properly formatted
        if metadatas is None:
            metadatas = [{} for _ in documents]
        elif len(metadatas) != len(documents):
            print(f"Warning: Number of metadata items ({len(metadatas)}) does not match number of documents ({len(documents)})")
            return False
            
        try:
            self.collection.add(
                ids=doc_ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            return True
        except Exception as e:
            print(f"Error adding documents to RAG: {e}")
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
