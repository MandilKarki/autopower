# Test script for using llama3.2 model with local Ollama server
import os
import time
from intelligent_report.intelligent_report import IntelligentReportGenerator
from intelligent_report.ai_utils import AIManager

def test_llama3_model():
    print("\n===== Testing Llama3.2 with Local Ollama Server =====\n")
    
    # Initialize the AI manager directly to test connection
    print("Initializing AI Manager with llama3.2...")
    ai_manager = AIManager()
    
    # Simple prompt to test generation capabilities
    print("\nGenerating sample content with llama3.2...")
    prompt = "Write a short paragraph about artificial intelligence and its impact on report generation."
    
    # Measure response time
    start_time = time.time()
    response = ai_manager.generate_content(prompt)
    end_time = time.time()
    
    print(f"\nGeneration time: {end_time - start_time:.2f} seconds")
    print("\nGenerated content:")
    print("-" * 50)
    print(response)
    print("-" * 50)
    
    # Create a simple report using the updated configuration
    print("\nCreating a simple report with the Intelligent Report Generator...")
    report = IntelligentReportGenerator("Llama3.2 Test Report")
    
    # Add a generated section
    print("\nGenerating AI section for the report...")
    report.generate_ai_section(
        "AI in Business",
        "The role of large language models in modern business analytics",
        detailed=True
    )
    
    # Generate and save the report
    output_path = "output/llama3_test_report.md"
    print(f"\nSaving report to {output_path}...")
    report.generate_report(output_path)
    
    print(f"\nTest completed. Report saved to {output_path}")

if __name__ == "__main__":
    test_llama3_model()
