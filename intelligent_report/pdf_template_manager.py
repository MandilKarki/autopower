# PDF Template Manager for the Intelligent Report Generator
import os
import time
from typing import Dict, List, Any, Optional, Union
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from io import BytesIO

class PDFTemplateManager:
    """
    Manages PDF templates and fills them with content from the report
    """
    
    def __init__(self, template_dir: str = "pdf_templates"):
        """
        Initialize the PDF Template Manager
        
        Args:
            template_dir: Directory containing PDF templates
        """
        self.template_dir = template_dir
        self.templates = {}
        
        # Create template directory if it doesn't exist
        os.makedirs(self.template_dir, exist_ok=True)
        
        # Load available templates
        self._load_templates()
    
    def _load_templates(self):
        """
        Load available PDF templates from the template directory
        """
        if not os.path.exists(self.template_dir):
            return
            
        for file in os.listdir(self.template_dir):
            if file.lower().endswith(".pdf"):
                template_path = os.path.join(self.template_dir, file)
                template_name = os.path.splitext(file)[0]
                self.templates[template_name] = template_path
    
    def list_templates(self) -> List[str]:
        """
        List available PDF templates
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
    
    def create_empty_template(self, template_name: str, sections: List[str]) -> str:
        """
        Create an empty PDF template with placeholders for sections
        
        Args:
            template_name: Name of the template
            sections: List of section names to include
            
        Returns:
            Path to the created template
        """
        # Create a PDF file
        template_path = os.path.join(self.template_dir, f"{template_name}.pdf")
        
        # Create the document
        doc = SimpleDocTemplate(template_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create elements for the PDF
        elements = []
        
        # Title
        title_style = styles["Title"]
        elements.append(Paragraph(f"{{title}}", title_style))
        elements.append(Spacer(1, 12))
        
        # Subtitle
        subtitle_style = styles["Heading2"]
        elements.append(Paragraph(f"{{subtitle}}", subtitle_style))
        elements.append(Spacer(1, 24))
        
        # Add sections
        for section in sections:
            # Section header
            section_style = styles["Heading1"]
            elements.append(Paragraph(section, section_style))
            elements.append(Spacer(1, 12))
            
            # Section content placeholder
            placeholder = f"{{{section.lower().replace(' ', '_')}}}"
            content_style = styles["Normal"]
            elements.append(Paragraph(placeholder, content_style))
            elements.append(Spacer(1, 24))
        
        # Build the PDF
        doc.build(elements)
        
        # Add to templates dict
        self.templates[template_name] = template_path
        
        return template_path
    
    def generate_pdf_from_template(self, template_name: str, content: Dict[str, str], output_path: str) -> str:
        """
        Generate a PDF from a template with the provided content
        
        Args:
            template_name: Name of the template to use
            content: Dictionary mapping placeholders to content
            output_path: Path to save the generated PDF
            
        Returns:
            Path to the generated PDF
        """
        if template_name not in self.templates:
            raise ValueError(f"Template {template_name} not found")
            
        template_path = self.templates[template_name]
        
        # Create the document
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = styles["Title"]
        heading1_style = styles["Heading1"]
        heading2_style = styles["Heading2"]
        normal_style = styles["Normal"]
        
        # Create elements for the PDF
        elements = []
        
        # Title
        title = content.get("title", "Untitled Report")
        elements.append(Paragraph(title, title_style))
        elements.append(Spacer(1, 12))
        
        # Subtitle
        subtitle = content.get("subtitle", "Generated Report")
        elements.append(Paragraph(subtitle, heading2_style))
        elements.append(Spacer(1, 24))
        
        # Add sections
        for key, value in content.items():
            # Skip title and subtitle
            if key in ["title", "subtitle"]:
                continue
                
            # Convert snake_case to Title Case for headings
            section_title = key.replace("_", " ").title()
            
            # Section header
            elements.append(Paragraph(section_title, heading1_style))
            elements.append(Spacer(1, 12))
            
            # Section content
            # Split paragraphs and add them separately
            paragraphs = value.split("\n\n")
            for para in paragraphs:
                if para.strip():
                    elements.append(Paragraph(para, normal_style))
                    elements.append(Spacer(1, 10))
            
            elements.append(Spacer(1, 24))
        
        # Build the PDF
        doc.build(elements)
        
        return output_path
    
    def generate_pdf_from_sections(self, sections: List[Dict], title: str, subtitle: str, output_path: str) -> str:
        """
        Generate a PDF directly from section data
        
        Args:
            sections: List of section dictionaries with title and content
            title: Report title
            subtitle: Report subtitle
            output_path: Path to save the generated PDF
            
        Returns:
            Path to the generated PDF
        """
        # Create content dictionary
        content = {
            "title": title,
            "subtitle": subtitle
        }
        
        # Add sections
        for section in sections:
            key = section.title.lower().replace(" ", "_")
            content[key] = section.content
        
        # Create a temporary template name
        temp_template_name = f"temp_{int(time.time())}"
        section_names = [section.title for section in sections]
        
        # Create template
        self.create_empty_template(temp_template_name, section_names)
        
        # Generate PDF
        result = self.generate_pdf_from_template(temp_template_name, content, output_path)
        
        # Clean up temporary template
        if temp_template_name in self.templates:
            temp_path = self.templates[temp_template_name]
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            del self.templates[temp_template_name]
        
        return result
