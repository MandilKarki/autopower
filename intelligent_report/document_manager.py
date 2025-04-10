# Document Management System for RAG
import os
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

class DocumentManager:
    """
    Manages document collections for RAG processing, allowing multiple contexts
    """
    
    def __init__(self, base_dir: str = "documents"):
        """
        Initialize the document manager
        
        Args:
            base_dir: Base directory for storing document references
        """
        self.base_dir = base_dir
        self.collections = {}
        self.active_collection = None
        
        # Create base directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Load existing collections
        self._load_collections()
    
    def _load_collections(self):
        """
        Load existing document collections from metadata files
        """
        collections_path = os.path.join(self.base_dir, "collections.json")
        if os.path.exists(collections_path):
            try:
                with open(collections_path, 'r') as f:
                    self.collections = json.load(f)
                # Set active collection if any exist
                if self.collections and not self.active_collection:
                    self.active_collection = next(iter(self.collections))
            except Exception as e:
                print(f"Error loading document collections: {e}")
                self.collections = {}
    
    def _save_collections(self):
        """
        Save current collection metadata
        """
        collections_path = os.path.join(self.base_dir, "collections.json")
        try:
            with open(collections_path, 'w') as f:
                json.dump(self.collections, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving document collections: {e}")
            return False
    
    def create_collection(self, name: str, description: str = "") -> bool:
        """
        Create a new document collection
        
        Args:
            name: Name of the collection
            description: Optional description
            
        Returns:
            bool: Success or failure
        """
        if name in self.collections:
            print(f"Collection '{name}' already exists")
            return False
        
        self.collections[name] = {
            "description": description,
            "documents": [],
            "created_at": str(Path().absolute()),
        }
        
        # Set as active if first collection
        if not self.active_collection:
            self.active_collection = name
        
        self._save_collections()
        return True
    
    def switch_collection(self, name: str) -> bool:
        """
        Switch to a different document collection
        
        Args:
            name: Name of the collection to switch to
            
        Returns:
            bool: Success or failure
        """
        if name not in self.collections:
            print(f"Collection '{name}' does not exist")
            return False
        
        self.active_collection = name
        print(f"Switched to collection: {name}")
        return True
    
    def add_document(self, path: str, metadata: Optional[Dict] = None) -> bool:
        """
        Add a document to the active collection
        
        Args:
            path: Path to the document
            metadata: Optional metadata about the document
            
        Returns:
            bool: Success or failure
        """
        if not self.active_collection:
            print("No active collection. Create or select a collection first.")
            return False
        
        if not os.path.exists(path):
            print(f"Document not found: {path}")
            return False
        
        # Check if document already exists in collection
        documents = self.collections[self.active_collection]["documents"]
        for doc in documents:
            if doc["path"] == path:
                print(f"Document already exists in collection: {path}")
                return False
        
        # Add document to collection
        doc_info = {
            "path": path,
            "name": os.path.basename(path),
            "metadata": metadata or {}
        }
        
        self.collections[self.active_collection]["documents"].append(doc_info)
        self._save_collections()
        return True
    
    def remove_document(self, path: str) -> bool:
        """
        Remove a document from the active collection
        
        Args:
            path: Path to the document
            
        Returns:
            bool: Success or failure
        """
        if not self.active_collection:
            print("No active collection.")
            return False
        
        documents = self.collections[self.active_collection]["documents"]
        original_count = len(documents)
        
        # Filter out the document
        self.collections[self.active_collection]["documents"] = [
            doc for doc in documents if doc["path"] != path
        ]
        
        # Check if any document was removed
        if len(self.collections[self.active_collection]["documents"]) == original_count:
            print(f"Document not found in collection: {path}")
            return False
        
        self._save_collections()
        return True
    
    def list_collections(self) -> List[Dict]:
        """
        Get a list of all document collections
        
        Returns:
            List of collection information
        """
        result = []
        for name, info in self.collections.items():
            result.append({
                "name": name,
                "description": info["description"],
                "document_count": len(info["documents"]),
                "is_active": name == self.active_collection
            })
        return result
    
    def list_documents(self, collection_name: Optional[str] = None) -> List[Dict]:
        """
        List all documents in a collection
        
        Args:
            collection_name: Name of collection (uses active collection if None)
            
        Returns:
            List of document information
        """
        name = collection_name or self.active_collection
        if not name or name not in self.collections:
            return []
        
        return self.collections[name]["documents"]
    
    def clear_collection(self, name: Optional[str] = None) -> bool:
        """
        Clear all documents from a collection
        
        Args:
            name: Name of collection (uses active collection if None)
            
        Returns:
            bool: Success or failure
        """
        collection_name = name or self.active_collection
        if not collection_name or collection_name not in self.collections:
            print(f"Collection not found: {collection_name}")
            return False
        
        self.collections[collection_name]["documents"] = []
        self._save_collections()
        return True
    
    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection
        
        Args:
            name: Name of the collection to delete
            
        Returns:
            bool: Success or failure
        """
        if name not in self.collections:
            print(f"Collection not found: {name}")
            return False
        
        # Remove the collection
        del self.collections[name]
        
        # Update active collection if needed
        if self.active_collection == name:
            self.active_collection = next(iter(self.collections)) if self.collections else None
        
        self._save_collections()
        return True
    
    def get_active_documents(self) -> List[str]:
        """
        Get paths of all documents in the active collection
        
        Returns:
            List of document paths
        """
        if not self.active_collection:
            return []
        
        return [doc["path"] for doc in self.collections[self.active_collection]["documents"]]
    
    def get_collection_info(self, name: Optional[str] = None) -> Dict:
        """
        Get information about a collection
        
        Args:
            name: Name of collection (uses active collection if None)
            
        Returns:
            Collection information
        """
        collection_name = name or self.active_collection
        if not collection_name or collection_name not in self.collections:
            return {}
        
        info = self.collections[collection_name].copy()
        info["name"] = collection_name
        info["document_count"] = len(info["documents"])
        return info
