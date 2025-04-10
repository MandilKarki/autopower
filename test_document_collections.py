# Test script for document collection management with the Intelligent Report Generator
import os
from intelligent_report.intelligent_report import IntelligentReportGenerator

def create_test_documents():
    """Create some simple test documents in different folders"""
    # Create directories
    os.makedirs("test_docs/financial", exist_ok=True)
    os.makedirs("test_docs/technical", exist_ok=True)
    
    # Financial documents
    with open("test_docs/financial/market_analysis.txt", "w") as f:
        f.write("""Market Analysis 2024
        
        The global market has seen significant growth in technology sectors, with AI-related 
        stocks showing a 25% increase year over year. Financial services are integrating 
        blockchain technologies at an unprecedented rate, leading to increased efficiency 
        and reduced transaction costs.
        
        Key economic indicators suggest moderate inflation of 2.8% across major economies,
        with interest rates stabilizing in the 3-4% range for most developed nations.
        """)
    
    with open("test_docs/financial/investment_strategy.txt", "w") as f:
        f.write("""Investment Strategy Guide
        
        Diversification remains the cornerstone of sound investment strategy. In the 
        current market, allocating 40% to equity, 30% to fixed income, 15% to 
        alternative investments, and 15% to cash equivalents provides optimal 
        risk-adjusted returns based on historical modeling.
        
        For growth-oriented portfolios, increasing exposure to emerging market 
        equities and technology sector ETFs may provide enhanced returns, albeit 
        with increased volatility.""")
    
    # Technical documents
    with open("test_docs/technical/machine_learning.txt", "w") as f:
        f.write("""Machine Learning Fundamentals
        
        Machine learning algorithms can be broadly categorized into supervised learning, 
        unsupervised learning, and reinforcement learning. Supervised learning requires 
        labeled data and is used for classification and regression tasks. Common algorithms 
        include linear regression, decision trees, and neural networks.
        
        Unsupervised learning works with unlabeled data and is used for clustering and 
        dimensionality reduction. K-means clustering and principal component analysis (PCA) 
        are frequently used unsupervised learning techniques.""")
    
    with open("test_docs/technical/cloud_computing.txt", "w") as f:
        f.write("""Cloud Computing Architecture
        
        Modern cloud architectures typically consist of three service models: Infrastructure 
        as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS). 
        
        Containerization technologies like Docker have revolutionized application deployment, 
        while orchestration platforms such as Kubernetes manage the scaling and operation of 
        containerized applications across distributed environments.
        
        Serverless computing represents a further evolution, allowing developers to build and 
        run applications without managing servers, typically on a pay-per-execution model.""")
    
    return {
        "financial": ["test_docs/financial/market_analysis.txt", "test_docs/financial/investment_strategy.txt"],
        "technical": ["test_docs/technical/machine_learning.txt", "test_docs/technical/cloud_computing.txt"]
    }

def test_document_collections():
    """Test document collection management with the Intelligent Report Generator"""
    print("\n===== Testing Document Collection Management =====\n")
    
    # Create test documents
    print("Creating test documents...")
    doc_paths = create_test_documents()
    
    # Initialize the report generator
    print("\nInitializing Intelligent Report Generator...")
    report = IntelligentReportGenerator("Document Collection Test Report")
    
    # Create document collections
    print("\nCreating document collections...")
    report.create_document_collection("financial", "Financial documents for market analysis")
    report.create_document_collection("technical", "Technical documents on computing topics")
    
    # List collections
    print("\nListing available document collections:")
    collections = report.list_document_collections()
    for collection in collections:
        print(f"- {collection['name']}: {collection['description']} ({collection['document_count']} documents)")
    
    # Add documents to the financial collection
    print("\nAdding documents to the financial collection...")
    report.switch_document_collection("financial")
    for doc_path in doc_paths["financial"]:
        report.ingest_document(doc_path)
    
    # Generate a report section using the financial collection
    print("\nGenerating a report section with the financial collection...")
    report.rag_query("Financial Market Overview", "What are the current trends in the financial market?")
    
    # Now switch to the technical collection
    print("\nSwitching to the technical collection...")
    report.switch_document_collection("technical")
    
    # Add documents to the technical collection
    print("Adding documents to the technical collection...")
    for doc_path in doc_paths["technical"]:
        report.ingest_document(doc_path)
    
    # Generate a report section using the technical collection
    print("\nGenerating a report section with the technical collection...")
    report.rag_query("Technical Computing Overview", "Explain the key concepts in cloud computing architecture")
    
    # Generate the final report with sections from both collections
    output_path = "output/document_collection_test.md"
    print(f"\nGenerating final report to {output_path}...")
    report.generate_report(output_path)
    
    print(f"\nTest completed. Report saved to {output_path}")
    
    # List the document in each collection
    print("\nDocuments in financial collection:")
    report.switch_document_collection("financial")
    financial_docs = report.list_documents()
    for doc in financial_docs:
        print(f"- {doc['name']}")
    
    print("\nDocuments in technical collection:")
    report.switch_document_collection("technical")
    technical_docs = report.list_documents()
    for doc in technical_docs:
        print(f"- {doc['name']}")

if __name__ == "__main__":
    test_document_collections()
