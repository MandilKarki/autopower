import os
import sys
from intelligent_report.pdf_template_manager import PDFTemplateManager

def main():
    # Initialize the PDF Template Manager
    template_manager = PDFTemplateManager("pdf_templates")
    
    # Make sure the template directory exists
    os.makedirs("pdf_templates", exist_ok=True)
    
    # Create a sample report template
    template_name = "standard_report"
    sections = [
        "Executive Summary",
        "Introduction",
        "Key Findings",
        "Data Analysis",
        "Recommendations",
        "Conclusion"
    ]
    
    # Create the template
    template_path = template_manager.create_empty_template(template_name, sections)
    print(f"Created template: {template_path}")
    
    # Now let's generate a sample PDF using this template
    sample_content = {
        "title": "Sample Report Title",
        "subtitle": "Generated with PDF Template System",
        "executive_summary": "This is an executive summary of the report. It provides a brief overview of the key points covered in the report.",
        "introduction": "This introduction section explains the purpose and scope of the report. It sets the context for the reader.",
        "key_findings": "This section highlights the most significant findings from the analysis. Several important trends were identified...",
        "data_analysis": "In this section, we present a detailed analysis of the data. The analysis shows that...",
        "recommendations": "Based on our analysis, we recommend the following actions: 1) First recommendation, 2) Second recommendation, 3) Third recommendation.",
        "conclusion": "In conclusion, this report has demonstrated several key insights that can be used to inform future decision-making."
    }
    
    # Generate the PDF
    output_path = "output/sample_report.pdf"
    pdf_path = template_manager.generate_pdf_from_template(template_name, sample_content, output_path)
    print(f"Generated PDF from template: {pdf_path}")
    
    # Create a business report template
    business_template = "business_report"
    business_sections = [
        "Market Overview",
        "Financial Analysis",
        "Competitive Landscape",
        "Growth Opportunities",
        "Risk Assessment",
        "Strategic Recommendations"
    ]
    
    # Create the business template
    business_template_path = template_manager.create_empty_template(business_template, business_sections)
    print(f"Created business template: {business_template_path}")
    
    print("\nAvailable templates:")
    for template in template_manager.list_templates():
        print(f" - {template}")

if __name__ == "__main__":
    main()
