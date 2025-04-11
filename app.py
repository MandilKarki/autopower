import os
import base64
import tempfile
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from ppt_generator import PowerPointGenerator

# Create necessary directories
os.makedirs('pdf_templates', exist_ok=True)
os.makedirs('datasources', exist_ok=True)
os.makedirs('output', exist_ok=True)

# Set page config
st.set_page_config(page_title="AI PowerPoint Generator", page_icon="📊", layout="wide")

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        color: #0D47A1;
        margin-top: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1976D2;
        color: white;
        padding: 0.5rem;
    }
    .file-upload-label {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Main app title
st.markdown("<h1 class='main-header'>AI PowerPoint Generator</h1>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "PDF Template Setup", "Data Source Management", "Generate PowerPoint"])

# Function to create a download link for files
def get_download_link(file_path, link_text):
    with open(file_path, "rb") as file:
        contents = file.read()
    b64 = base64.b64encode(contents).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.presentationml.presentation;base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
    return href

# Home page
if page == "Home":
    st.markdown("<h2 class='section-header'>Welcome to AI PowerPoint Generator</h2>", unsafe_allow_html=True)
    st.write("""
    This tool helps you create professional PowerPoint presentations using AI technology. 
    You can use PDF templates, data sources, and AI to generate content-rich presentations.
    
    The generator offers the following features:
    - Upload PDF templates for the AI to populate with information
    - Generate graphs and charts based on Excel or CSV datasets
    - Extract context from PDF files to generate content
    - Customize parameters through this user interface
    """)
    
    st.markdown("<h3 class='section-header'>Getting Started</h3>", unsafe_allow_html=True)
    st.write("""
    1. Go to the **PDF Template Setup** page to upload your template PDFs
    2. Visit the **Data Source Management** page to upload your data files (Excel, CSV) and context PDFs
    3. Use the **Generate PowerPoint** page to create your presentation
    """)
    
    # Display example image if available
    example_path = os.path.join("assets", "example.png")
    if os.path.exists(example_path):
        st.image(example_path, caption="Example PowerPoint Generated with AI")

# PowerPoint Template Setup Page
elif page == "PDF Template Setup":
    st.markdown("<h2 class='section-header'>PowerPoint Template Management</h2>", unsafe_allow_html=True)
    st.write("""
    Upload PowerPoint templates (.pptx) that will serve as the structure for your presentations.
    The AI will analyze these templates and use them to generate slides with appropriate content.
    """)
    
    # Upload new template
    st.markdown("<p class='file-upload-label'>Upload a new PowerPoint template:</p>", unsafe_allow_html=True)
    uploaded_template = st.file_uploader("Choose a PowerPoint file", type="pptx", key="template_uploader")
    template_name = st.text_input("Template Name (optional)")
    
    if st.button("Save Template"):
        if uploaded_template is not None:
            # Create a name for the template if not provided
            if not template_name:
                template_name = uploaded_template.name
            else:
                # Ensure the file has a .pptx extension
                if not template_name.lower().endswith(".pptx"):
                    template_name += ".pptx"
            
            # Save the uploaded file
            template_path = os.path.join("pdf_templates", template_name)
            with open(template_path, "wb") as f:
                f.write(uploaded_template.getbuffer())
            
            st.success(f"Template '{template_name}' saved successfully!")
        else:
            st.error("Please upload a PowerPoint file first.")
    
    # Display existing templates
    st.markdown("<h3 class='section-header'>Existing Templates</h3>", unsafe_allow_html=True)
    templates = [f for f in os.listdir("pdf_templates") if f.lower().endswith(".pptx")]
    
    if templates:
        for template in templates:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(template)
            with col2:
                if st.button("Preview", key=f"preview_{template}"):
                    template_path = os.path.join("pdf_templates", template)
                    # For PowerPoint files, we can't easily preview them directly,
                    # so just show a message
                    st.info(f"PowerPoint preview for {template} is not available in the browser. The file has been saved successfully.")
            with col3:
                if st.button("Delete", key=f"delete_{template}"):
                    os.remove(os.path.join("pdf_templates", template))
                    st.success(f"Template '{template}' deleted successfully!")
                    st.experimental_rerun()
    else:
        st.info("No templates available. Please upload a PowerPoint template (.pptx file).")

# Data Source Management Page
elif page == "Data Source Management":
    st.markdown("<h2 class='section-header'>Data Source Management</h2>", unsafe_allow_html=True)
    st.write("""
    Upload data files (Excel/CSV) for chart generation and PDF files for context extraction.
    These sources will be used to enrich your PowerPoint presentations with data-driven visuals and content.
    """)
    
    # Data source tabs
    tab1, tab2 = st.tabs(["Data Files (Excel/CSV)", "Context PDFs"])
    
    # Data Files (Excel/CSV) Tab
    with tab1:
        st.markdown("<p class='file-upload-label'>Upload a data file:</p>", unsafe_allow_html=True)
        uploaded_data = st.file_uploader("Choose Excel or CSV file", type=["xlsx", "xls", "csv"], key="data_uploader")
        data_name = st.text_input("Data File Name (optional)", key="data_name")
        
        if st.button("Save Data File"):
            if uploaded_data is not None:
                # Create a name for the data file if not provided
                if not data_name:
                    data_name = uploaded_data.name
                else:
                    # Ensure the file has the appropriate extension
                    if uploaded_data.name.lower().endswith(".csv") and not data_name.lower().endswith(".csv"):
                        data_name += ".csv"
                    elif (uploaded_data.name.lower().endswith(".xlsx") or uploaded_data.name.lower().endswith(".xls")) and not (data_name.lower().endswith(".xlsx") or data_name.lower().endswith(".xls")):
                        data_name += ".xlsx"
                
                # Save the uploaded file
                data_path = os.path.join("datasources", data_name)
                with open(data_path, "wb") as f:
                    f.write(uploaded_data.getbuffer())
                
                st.success(f"Data file '{data_name}' saved successfully!")
            else:
                st.error("Please upload a data file first.")
        
        # Display existing data files
        st.markdown("<h3 class='section-header'>Existing Data Files</h3>", unsafe_allow_html=True)
        data_files = [f for f in os.listdir("datasources") if f.lower().endswith((".xlsx", ".xls", ".csv"))]
        
        if data_files:
            for data_file in data_files:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(data_file)
                with col2:
                    if st.button("Preview", key=f"preview_{data_file}"):
                        data_path = os.path.join("datasources", data_file)
                        # Read and display data preview
                        if data_file.lower().endswith(".csv"):
                            df = pd.read_csv(data_path)
                        else:
                            df = pd.read_excel(data_path)
                        st.dataframe(df.head())
                with col3:
                    if st.button("Delete", key=f"delete_{data_file}"):
                        os.remove(os.path.join("datasources", data_file))
                        st.success(f"Data file '{data_file}' deleted successfully!")
                        st.experimental_rerun()
        else:
            st.info("No data files available. Please upload an Excel or CSV file.")
    
    # Context PDFs Tab
    with tab2:
        st.markdown("<p class='file-upload-label'>Upload a context PDF:</p>", unsafe_allow_html=True)
        uploaded_context = st.file_uploader("Choose a PDF file", type="pdf", key="context_uploader")
        context_name = st.text_input("Context PDF Name (optional)", key="context_name")
        
        if st.button("Save Context PDF"):
            if uploaded_context is not None:
                # Create a name for the context PDF if not provided
                if not context_name:
                    context_name = uploaded_context.name
                else:
                    # Ensure the file has a .pdf extension
                    if not context_name.lower().endswith(".pdf"):
                        context_name += ".pdf"
                
                # Save the uploaded file
                context_path = os.path.join("datasources", context_name)
                with open(context_path, "wb") as f:
                    f.write(uploaded_context.getbuffer())
                
                st.success(f"Context PDF '{context_name}' saved successfully!")
            else:
                st.error("Please upload a PDF file first.")
        
        # Display existing context PDFs
        st.markdown("<h3 class='section-header'>Existing Context PDFs</h3>", unsafe_allow_html=True)
        context_pdfs = [f for f in os.listdir("datasources") if f.lower().endswith(".pdf")]
        
        if context_pdfs:
            for context_pdf in context_pdfs:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(context_pdf)
                with col2:
                    if st.button("Preview", key=f"preview_{context_pdf}"):
                        context_path = os.path.join("datasources", context_pdf)
                        # Display PDF preview
                        with open(context_path, "rb") as f:
                            pdf_bytes = f.read()
                        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="500" type="application/pdf"></iframe>'
                        st.markdown(pdf_display, unsafe_allow_html=True)
                with col3:
                    if st.button("Delete", key=f"delete_{context_pdf}"):
                        os.remove(os.path.join("datasources", context_pdf))
                        st.success(f"Context PDF '{context_pdf}' deleted successfully!")
                        st.experimental_rerun()
        else:
            st.info("No context PDFs available. Please upload a PDF file.")

# Generate PowerPoint Page
elif page == "Generate PowerPoint":
    st.markdown("<h2 class='section-header'>Generate PowerPoint Presentation</h2>", unsafe_allow_html=True)
    st.write("""
    Configure your PowerPoint generation settings and create your presentation.
    Combine PowerPoint templates, data sources, and AI-generated content with RAG to create professional slides.
    """)
    
    # Configure PowerPoint Generation
    st.markdown("<h3 class='section-header'>Configuration</h3>", unsafe_allow_html=True)
    
    # Presentation title and output name
    ppt_title = st.text_input("Presentation Title", "AI Generated Presentation")
    output_name = st.text_input("Output File Name", "my_presentation.pptx")
    
    # Template selection
    templates = [f for f in os.listdir("pdf_templates") if f.lower().endswith(".pptx")]
    template_option = st.selectbox("Select PowerPoint Template", ["None"] + templates if templates else ["None"])
    
    # Data source selection for charts
    data_files = [f for f in os.listdir("datasources") if f.lower().endswith((".xlsx", ".xls", ".csv"))]
    data_option = st.selectbox("Select Data Source for Charts", ["None"] + data_files if data_files else ["None"])
    
    # Context PDF selection
    context_pdfs = [f for f in os.listdir("datasources") if f.lower().endswith(".pdf")]
    context_option = st.selectbox("Select Context PDF", ["None"] + context_pdfs if context_pdfs else ["None"])
    
    # RAG Options
    st.markdown("<h3 class='section-header'>RAG Options</h3>", unsafe_allow_html=True)
    use_rag = st.checkbox("Use RAG for Content Generation", True)
    
    # Model selection
    ollama_models = ["llama3.2:latest", "llama2:latest", "mistral:latest"]
    selected_model = st.selectbox("Select Ollama Model", options=ollama_models, index=0, disabled=not use_rag)
    
    # Advanced RAG options in expander
    with st.expander("Advanced RAG Options"):
        chunk_size = st.slider("PDF Chunk Size", 500, 2500, 1000, 100, disabled=not use_rag)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50, disabled=not use_rag)
        n_results = st.slider("Number of Results per Query", 1, 10, 5, disabled=not use_rag)
    
    # Chart Generation options
    st.markdown("<h3 class='section-header'>Chart Generation Options</h3>", unsafe_allow_html=True)
    include_charts = st.checkbox("Include Charts", True)
    
    # Advanced chart options in expander
    with st.expander("Advanced Chart Options"):
        chart_types = st.multiselect("Chart Types to Include", 
                                   ["bar", "column", "line", "pie"], 
                                   default=["column", "pie"])
        include_data_summary = st.checkbox("Include Data Summary Slide", True)
        include_table_slide = st.checkbox("Include Data Table Slide", True)
    
    # Generate PowerPoint button
    if st.button("Generate PowerPoint"):
        with st.spinner("Generating PowerPoint presentation..."):
            try:
                # Initialize PowerPoint generator
                template_path = os.path.join("pdf_templates", template_option) if template_option != "None" else None
                ppt_gen = PowerPointGenerator(template_path)
                
                # Ensure proper output file name
                if not output_name.lower().endswith(".pptx"):
                    output_name += ".pptx"
                output_path = os.path.join("output", output_name)
                
                # Process context PDF with RAG if selected
                if context_option != "None" and use_rag:
                    context_path = os.path.join("datasources", context_option)
                    st.info(f"Processing {context_option} with RAG...")
                    
                    # Custom RAG parameters if specified
                    if 'chunk_size' in locals() and 'chunk_overlap' in locals():
                        chunks_added = ppt_gen.rag_processor.process_pdf(context_path, chunk_size, chunk_overlap)
                    else:
                        chunks_added = ppt_gen.rag_processor.process_pdf(context_path)
                    
                    st.info(f"Added {chunks_added} chunks to the vector database for RAG.")
                
                # Create AI content dictionary with title
                ai_content = {'title': ppt_title}
                
                # Process based on selected options
                if template_option != "None" and context_option != "None":
                    # Generate from template and context PDF using RAG
                    context_path = os.path.join("datasources", context_option)
                    
                    # Generate the presentation using RAG with selected model
                    # Create a PowerPointGenerator instance with the selected model
                    ppt_gen = PowerPointGenerator(template_path, model=selected_model)
                    ppt_gen.generate_from_template(template_path, context_path, output_path, ai_content)
                    generation_type = "PowerPoint template and RAG with context PDF"
                
                elif context_option != "None" and use_rag:
                    # Generate presentation just from context PDF using RAG
                    context_path = os.path.join("datasources", context_option)
                    
                    # Generate common sections using RAG
                    sections = ["Introduction", "Key Points", "Analysis", "Conclusion"]
                    for section in sections:
                        query = f"Generate content for the {section} section of the presentation based on the document."
                        content = ppt_gen.rag_processor.generate_content_with_rag(query, model=selected_model)
                        ppt_gen.generate_slide_from_text(section, content)
                    
                    # Save the presentation
                    ppt_gen.save(output_path)
                    generation_type = "RAG with context PDF"
                
                elif data_option != "None":
                    # Generate from data file
                    data_path = os.path.join("datasources", data_option)
                    ppt_gen.generate_from_data(data_path, output_path, ppt_title, include_charts)
                    generation_type = "data analysis"
                
                else:
                    # Generate a basic presentation
                    # Create a title slide
                    slide_layout = ppt_gen.presentation.slide_layouts[0]  # Title slide layout
                    slide = ppt_gen.presentation.slides.add_slide(slide_layout)
                    slide.shapes.title.text = ppt_title
                    
                    # Add some basic content slides
                    ppt_gen.generate_slide_from_text("Introduction", "This is a basic presentation generated without a template or data source.")
                    ppt_gen.generate_slide_from_text("Content", "Add your content here. This slide was generated automatically.")
                    ppt_gen.generate_slide_from_text("Conclusion", "Thank you for using the AI PowerPoint Generator!")
                    
                    # Save the presentation
                    ppt_gen.save(output_path)
                    generation_type = "basic template"
                
                st.success(f"PowerPoint presentation generated successfully from {generation_type}!")
                
                # Provide download link
                st.markdown(get_download_link(output_path, "Download PowerPoint"), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error generating PowerPoint: {str(e)}")

# Run the app with: streamlit run app.py
