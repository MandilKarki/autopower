# Main Intelligent Report Generator Class
import os
import json
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any, Optional, Union

# Import from other modules
from .ai_utils import AIManager, RAGProcessor
from .document_processor import DocumentProcessor
from .data_analysis import DataAnalyzer
from .visualization import VisualizationManager
from .section_manager import SectionManager
from .config import ReportConfig


class IntelligentReportGenerator:
    """
    Intelligent Report Generator with PowerPoint and PDF capabilities
    """
    
    def __init__(self, report_title: str = "Intelligent Report", model_name: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize the Intelligent Report Generator"""
        # Load configuration
        self.config = ReportConfig(config_path)
        
        # Override title if provided
        if report_title != "Intelligent Report":
            self.config.set("report", "title", report_title)
        else:
            report_title = self.config.get("report", "title", "Intelligent Report")
            
        self.report_title = report_title
        self.referenced_documents = []
        self.creation_date = datetime.now().strftime("%Y-%m-%d")
        
        # Use model from config if not provided
        if model_name is None:
            model_name = self.config.get("ai_model", "name", None)
        
        # Initialize components
        self.ai_manager = AIManager(model_name)
        self.doc_processor = DocumentProcessor()
        self.data_analyzer = DataAnalyzer()
        self.section_manager = SectionManager()
        
        # Initialize visualization manager with configured output directory
        viz_dir = self.config.get("output", "visualization_dir", "temp")
        self.viz_manager = VisualizationManager(viz_dir)
        
        self.rag_processor = RAGProcessor(self.ai_manager)
        
        # Create output directories if they don't exist
        os.makedirs("output", exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
    
    def add_user_section(self, section_title: str, content: str, order: Optional[int] = None) -> 'IntelligentReportGenerator':
        """Add a user-provided section to the report"""
        self.section_manager.add_section(
            title=section_title,
            content=content,
            section_type="user",
            order=order
        )
        return self
    
    def generate_ai_section(self, section_title: str, prompt: str, detailed: bool = False, 
                           order: Optional[int] = None) -> 'IntelligentReportGenerator':
        """Generate an AI-written section based on a prompt"""
        word_count = "400-600" if detailed else "200-300"
        full_prompt = f'''Write a professional, informative section about {prompt}.
        The content should be factual, well-structured, and approximately {word_count} words.
        Use a formal tone suitable for a business or academic report.
        Include relevant statistics or examples where appropriate.
        Organize the content with clear paragraph breaks.'''
        
        result = self.ai_manager.generate_content(full_prompt)
        
        # Store metadata for future reference
        metadata = {
            "prompt": prompt,
            "detailed": detailed,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.section_manager.add_section(
            title=section_title,
            content=result,
            section_type="ai",
            order=order,
            metadata=metadata
        )
        return self
    
    def suggest_topics(self, base_topic: str = "business analytics", count: int = 5) -> List[Dict]:
        """Suggest relevant topics for report sections based on a base topic"""
        prompt = f'''Suggest {count} specific, focused topics related to "{base_topic}" that would make good sections in a professional report.
        For each topic, provide: 
        1. A concise section title (3-6 words)
        2. A brief description of what this section should cover (1-2 sentences)
        
        Format your response as a JSON list with objects containing 'title' and 'description' fields.
        Example: [{{'title': 'Market Segmentation Analysis', 'description': 'Examination of key market segments and targeting strategies.'}}]'''
        
        result = self.ai_manager.generate_content(prompt)
        return self.ai_manager.parse_json_response(result)
    
    def load_data(self, file_path: str) -> 'IntelligentReportGenerator':
        """Load data from CSV, Excel, or JSON file for analysis and visualization"""
        self.data_analyzer.load_data(file_path)
        return self
    
    def add_visualization(self, title: str, data: List, chart_type: str = "bar", 
                         labels: Optional[List] = None, **kwargs) -> 'IntelligentReportGenerator':
        """Add a data visualization to the report using raw data"""
        self.viz_manager.create_visualization(title, data, chart_type, labels, **kwargs)
        return self
    
    def create_visualization_from_dataframe(self, title: str, chart_type: str, 
                                           columns: Optional[List[str]] = None, **kwargs) -> 'IntelligentReportGenerator':
        """Create visualization from loaded dataframe data"""
        if self.data_analyzer.dataset is None or self.data_analyzer.dataset.empty:
            print("Error: No dataset loaded. Use load_data() first.")
            return self
        
        self.viz_manager.create_visualization_from_dataframe(
            df=self.data_analyzer.dataset,
            title=title,
            chart_type=chart_type,
            columns=columns,
            **kwargs
        )
        return self
    
    def auto_visualize(self, count: int = 3) -> 'IntelligentReportGenerator':
        """Automatically generate the most meaningful visualizations from the dataset"""
        if self.data_analyzer.dataset is None or self.data_analyzer.dataset.empty:
            print("Error: No dataset loaded. Use load_data() first.")
            return self
            
        # Get suggested visualizations
        suggestions = self.data_analyzer.suggest_visualizations()
        if not suggestions:
            return self
            
        # Create the top N suggested visualizations
        for i, suggestion in enumerate(suggestions[:count]):
            self.create_visualization_from_dataframe(
                title=suggestion['title'],
                chart_type=suggestion['type'],
                columns=suggestion['columns'],
                description=suggestion['description']
            )
            
        return self
    
    def ingest_document(self, file_path: str) -> bool:
        """Process and add document to RAG system for querying"""
        try:
            if not os.path.exists(file_path):
                print(f"Error: File {file_path} does not exist.")
                return False
                
            # Extract text from the document
            text = self.doc_processor.extract_text(file_path)
            
            # Get chunking configuration
            chunk_size = self.config.get("rag", "chunk_size", 1000)
            chunk_overlap = self.config.get("rag", "chunk_overlap", 200)
            
            # Chunk text for the RAG processor
            chunks = self.doc_processor.chunk_text(text, chunk_size, chunk_overlap)
            
            # Add document to RAG system
            document_name = os.path.basename(file_path)
            success = self.rag_processor.add_documents(
                documents=chunks,
                metadatas=[{"source": document_name} for _ in chunks]
            )
            
            # Add to referenced_documents if successful
            if success:
                self.referenced_documents.append(file_path)
                
                # Save to config if needed
                if self.config.get("rag", "save_references", True):
                    references = self.config.get("rag", "references", [])
                    references.append(file_path)
                    self.config.set("rag", "references", references)
            
            return success
        except Exception as e:
            print(f"Error ingesting document: {e}")
            return False
    
    def load_data_from_file(self, file_path: str) -> bool:
        """Load data from a CSV, Excel, or JSON file for analysis"""
        success = self.data_analyzer.load_data(file_path)
        
        # Save to config if needed
        if success and self.config.get("data", "save_data_source", True):
            self.config.set("data", "last_data_file", file_path)
        
        return success
    
    def create_visualization_from_dataframe(
        self,
        title: str,
        chart_type: str = "bar",
        columns: Optional[List[str]] = None,
        description: str = ""
    ) -> Dict:
        """Create visualizations from pandas DataFrame"""
        if self.data_analyzer.dataset is None:
            return {"error": "No dataset loaded. Please load data first."}
        
        # Get the dataset
        df = self.data_analyzer.dataset
        
        # If columns not specified, try to use intelligent selection
        if columns is None:
            # Get suggested columns based on chart type
            columns = self.data_analyzer.suggest_columns_for_chart(chart_type)
            if not columns:
                return {"error": f"Could not automatically determine columns for {chart_type} chart"}
        
        # Create the visualization
        result = self.viz_manager.create_visualization_from_dataframe(
            df=df,
            title=title,
            chart_type=chart_type,
            columns=columns,
            description=description
        )
        
        # Save visualization details to config if enabled
        if self.config.get("visualizations", "save_history", True):
            viz_history = self.config.get("visualizations", "history", [])
            viz_info = {
                "title": title,
                "type": chart_type,
                "columns": columns,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            viz_history.append(viz_info)
            self.config.set("visualizations", "history", viz_history)
        
        return result
    
    def rag_query(self, section_title: str, query: str, order: Optional[int] = None) -> 'IntelligentReportGenerator':
        """Query ingested documents and add the results as a section"""
        if not self.referenced_documents:
            content = "No documents have been ingested for processing."
            self.section_manager.add_section(
                title=section_title,
                content=content,
                section_type="rag",
                order=order,
                metadata={"query": query}
            )
            return self
        
        # Query for relevant documents
        results = self.rag_processor.query(query)
        
        if 'error' in results or not results.get('documents', [[]])[0]:
            content = f"No relevant information found for: {query}"
            self.section_manager.add_section(
                title=section_title,
                content=content,
                section_type="rag",
                order=order,
                metadata={"query": query}
            )
            return self
        
        # Generate augmented response
        result = self.rag_processor.generate_augmented_response(
            query=query,
            context_docs=results['documents']
        )
        
        # Format sources
        sources = "\n".join([f"• {meta['source']}" for meta in results['metadatas'][0]])
        
        # Create section with citations
        content = f"{result}\n\nSources:\n{sources}"
        
        # Add to section manager
        self.section_manager.add_section(
            title=section_title,
            content=content,
            section_type="rag",
            order=order,
            metadata={
                "query": query,
                "sources": [meta['source'] for meta in results['metadatas'][0]],
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        )
        
        return self
    
    def _format_section(self, section) -> str:
        """Format a section for the markdown report"""
        result = f"# {section.title}\n\n"
        
        if section.type in ["user", "ai", "rag"]:
            result += section.content
        
        return result
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate the final report in Markdown format"""
        # Use output path from config if not provided
        if output_path is None:
            output_path = self.config.get("output", "markdown", "output/generated_report.md")
            
        # Compile the report content
        content = [f"# {self.report_title}\n"]
        
        # Get sections in order
        sections = self.section_manager.get_sections_in_order()
        
        # Add a table of contents if we have sections
        if sections:
            content.append("## Table of Contents\n")
            for i, section in enumerate(sections, 1):
                content.append(f"{i}. {section.title}\n")
            content.append("\n")
        
        # Add each section
        for section in sections:
            content.append(self._format_section(section))
            content.append("\n\n")
        
        # Add visualizations
        if self.viz_manager.visualizations:
            content.append("## Visualizations\n")
            for viz in self.viz_manager.visualizations:
                content.append(f"### {viz['title']}\n")
                # If there's a description, add it
                if viz.get('description'):
                    content.append(f"_{viz['description']}_\n\n")
                content.append(f"![{viz['title']}]({viz['filename']})\n\n")
        
        # Write to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        
        print(f"Report generated successfully: {output_path}")
        return output_path
    
    def generate_presentation(self, output_path: Optional[str] = None) -> str:
        """Generate a PowerPoint presentation based on the report data"""
        # Import here to avoid circular imports
        from .powerpoint_generator import PowerPointGenerator
        
        # Use output path from config if not provided
        if output_path is None:
            output_path = self.config.get("output", "powerpoint", "output/presentation.pptx")
        
        # Get PowerPoint title and other details from config
        subtitle = self.config.get("report", "subtitle", "Generated Report")
        author = self.config.get("report", "author", "")
        company = self.config.get("report", "company", "")
        
        # Create PowerPoint generator
        ppt_gen = PowerPointGenerator(self.report_title)
        ppt_gen.set_metadata(subtitle, author, company)
        
        # Add sections in order
        for section in self.section_manager.get_sections_in_order():
            ppt_gen.add_section(section.title, {
                "type": section.type,
                "content": section.content,
                "metadata": section.metadata or {}
            })
        
        # Add visualizations
        for viz in self.viz_manager.visualizations:
            ppt_gen.add_visualization(viz)
        
        # Generate the presentation
        return ppt_gen.generate(output_path)

    # Methods for working with templates and configs
    def load_template(self, template_path: str) -> bool:
        """Load a template configuration"""
        success = self.config.load_config(template_path)
        if success:
            # Update report title
            self.report_title = self.config.get("report", "title", self.report_title)
            
            # Create any predefined sections
            sections_config = self.config.get("sections", {})
            for section_id, section_info in sections_config.items():
                title = section_info.get("title", section_id)
                section_type = section_info.get("type", "user")
                order = section_info.get("order", None)
                
                # Handle different section types
                if section_type == "user" and "template" in section_info:
                    # Add user section with template content
                    self.add_user_section(title, section_info["template"], order)
                elif section_type == "ai" and "prompt" in section_info:
                    # Generate AI content based on prompt
                    detailed = section_info.get("detailed", False)
                    self.generate_ai_section(title, section_info["prompt"], detailed, order)
            
            # Auto-visualize if configured
            if self.config.get("visualizations", "auto_generate", False):
                auto_count = self.config.get("visualizations", "auto_count", 3)
                self.auto_visualize(auto_count)
            
            # Create custom visualizations if defined
            custom_viz = self.config.get("visualizations", "custom", [])
            for viz in custom_viz:
                if self.data_analyzer.dataset is not None and viz.get("type") and viz.get("columns"):
                    self.create_visualization_from_dataframe(
                        title=viz.get("title", "Visualization"),
                        chart_type=viz["type"],
                        columns=viz["columns"],
                        description=viz.get("description", "")
                    )
        
        return success
    
    def save_template(self, template_path: str) -> bool:
        """Save current configuration as a template"""
        # Update config with current settings
        self.config.set("report", "title", self.report_title)
        
        # Add sections to config
        # Clear existing sections first
        self.config.config["sections"] = {}
        
        # Add each section individually
        for section in self.section_manager.get_sections_in_order():
            section_id = section.title.lower().replace(" ", "_")
            section_config = {
                "title": section.title,
                "type": section.type,
                "order": section.order
            }
            
            # Add type-specific properties
            if section.type == "ai" and section.metadata and "prompt" in section.metadata:
                section_config["prompt"] = section.metadata["prompt"]
                section_config["detailed"] = section.metadata.get("detailed", False)
            elif section.type == "rag" and section.metadata and "query" in section.metadata:
                section_config["query"] = section.metadata["query"]
            
            # Add to config
            self.config.config["sections"][section_id] = section_config
        
        # Update visualization settings
        # Update existing visualization config or create a new one
        if "visualizations" not in self.config.config:
            self.config.config["visualizations"] = {}
            
        # Set auto-visualization settings    
        self.config.config["visualizations"]["auto_generate"] = self.config.get("visualizations", "auto_generate", True)
        self.config.config["visualizations"]["auto_count"] = self.config.get("visualizations", "auto_count", 3)
        
        # Set custom visualizations
        custom_viz = []
        for viz in self.viz_manager.visualizations:
            custom_viz.append({
                "title": viz["title"],
                "type": viz["chart_type"],
                "description": viz.get("description", ""),
                "columns": viz.get("columns", [])
            })
            
        self.config.config["visualizations"]["custom"] = custom_viz
        
        # Save to file
        return self.config.save_config(template_path)
