# Document processing utilities
import os
import re
import pdfplumber
from typing import List, Dict, Any, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer


class DocumentProcessor:
    """
    Handles document processing including text extraction and chunking
    """
    
    def __init__(self):
        pass
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from a document file"""
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            return ""
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Extract text based on file type
        if file_ext == '.pdf':
            return self._extract_text_from_pdf(file_path)
        elif file_ext in ['.txt', '.md', '.html', '.json']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            print(f"Warning: Unsupported file format {file_ext}")
            return ""
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks with overlap"""
        chunks = []
        
        # Split the text into chunks with overlap
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def extract_topics(self, text: str, num_topics: int = 5) -> List[str]:
        """Extract key topics from a document using TF-IDF"""
        # Clean and tokenize text
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        
        if len(sentences) < 5:
            # Not enough text for meaningful extraction
            return []
            
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(max_df=0.7, min_df=2, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Get top terms
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.sum(axis=0).A1
            top_indices = tfidf_scores.argsort()[-num_topics*2:][::-1]  # Get more than needed to filter
            top_terms = [feature_names[i] for i in top_indices]
            
            # Filter to more meaningful terms (longer words tend to be more meaningful)
            top_terms = [term for term in top_terms if len(term) > 3][:num_topics]
            
            return top_terms
        except Exception as e:
            print(f"Error extracting topics: {e}")
            return []
