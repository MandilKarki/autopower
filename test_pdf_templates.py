import os
import sys
from intelligent_report.intelligent_report import IntelligentReportGenerator

def main():
    # Initialize the Intelligent Report Generator
    report_generator = IntelligentReportGenerator(
        "PDF Template Test Report",
        model_name="llama3"
    )
    
    # Add some report sections
    report_generator.add_user_section(
        "Executive Summary", 
        "This report demonstrates the PDF template functionality of the Intelligent Report Generator."
    )
    
    report_generator.add_user_section(
        "Introduction", 
        "The Intelligent Report Generator now supports PDF templates that can be used to create consistently formatted reports."
    )
    
    report_generator.add_user_section(
        "Template System", 
        "The template system allows you to create reusable PDF templates with placeholders for different sections. \n\n"
        "These templates can be populated with content from the report generator, making it easy to create professional-looking PDFs."
    )
    
    report_generator.add_user_section(
        "How It Works", 
        "1. Create a template using the create_pdf_template method\n"
        "2. Add content to your report using the usual methods\n"
        "3. Generate a PDF using the generate_pdf_from_template method\n\n"
        "Alternatively, you can generate a PDF directly from your report sections without a template using the generate_pdf method."
    )
    
    # Create a basic template with the current sections
    print("Creating PDF template...")
    template_name = "report_template"
    template_path = report_generator.create_pdf_template(template_name)
    print(f"Created template at: {template_path}")
    
    # List available templates
    print("\nAvailable templates:")
    templates = report_generator.list_pdf_templates()
    for template in templates:
        print(f" - {template}")
    
    # Generate a PDF from the template
    print("\nGenerating PDF from template...")
    pdf_path = report_generator.generate_pdf_from_template(
        template_name, 
        "output/template_report.pdf"
    )
    print(f"Generated PDF at: {pdf_path}")
    
    # Generate a PDF directly
    print("\nGenerating PDF directly without template...")
    direct_pdf_path = report_generator.generate_pdf(
        "output/direct_report.pdf"
    )
    print(f"Generated direct PDF at: {direct_pdf_path}")
    
    print("\nPDF generation complete!")
    print("You can find the PDFs in the output directory.")

if __name__ == "__main__":
    main()
