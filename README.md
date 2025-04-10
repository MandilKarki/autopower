# Intelligent Report Generator

A comprehensive report generation system that creates both markdown reports and PowerPoint presentations using AI-generated content, data analysis, and automated visualizations.

## Features

- **AI Content Generation**: Uses Ollama models to generate high-quality report sections based on prompts
- **Template-Based Reports**: Create and save report templates for standardized formats
- **Topic Suggestions**: AI-powered suggestions for relevant report sections
- **Automatic Data Analysis**: Import and analyze data from CSV, Excel, or JSON files
- **Smart Visualization Generation**: Automatically creates charts based on data characteristics
- **RAG Capabilities**: Retrieval Augmented Generation for citing external documents in reports
- **PowerPoint Integration**: Generates professional PowerPoint presentations alongside markdown reports

## Setup

1. **Install Ollama**: Download from [Ollama's website](https://ollama.ai/) and install
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Optional**: Pull a specific model in Ollama (if you want to use a model other than the default):
   ```bash
   ollama pull mistral
   ```

## Usage

### Basic Usage

```python
from intelligent_report.intelligent_report import IntelligentReportGenerator

# Create a new report
report = IntelligentReportGenerator("My Analysis Report")

# Add content
report.add_user_section("Executive Summary", "This report analyzes...")
report.generate_ai_section("Industry Overview", "Current state of tech industry", detailed=True)

# Generate output
report.generate_report("output/report.md")
report.generate_presentation("output/presentation.pptx")
```

### Using Templates

```python
# Create a report from a template
report = IntelligentReportGenerator(config_path="templates/my_template.json")

# Load data for analysis
report.load_data_from_file("data/sample_data.csv")

# Generate output files (paths defined in template)
report.generate_report()
report.generate_presentation()

# Save as a new template for future use
report.save_template("templates/new_template.json")
```

## Components

- **AI Manager**: Controls interaction with AI models for content generation
- **Document Processor**: Extracts and processes text from various file formats
- **Data Analyzer**: Loads and analyzes datasets to extract insights
- **Visualization Manager**: Creates charts and graphs from data
- **Section Manager**: Organizes report sections with proper ordering
- **PowerPoint Generator**: Builds well-structured PowerPoint presentations

## Requirements

- Python 3.8+
- Ollama (for AI content generation)
- Required Python packages (see requirements.txt)

## License

MIT
