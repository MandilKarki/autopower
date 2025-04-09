# PowerPoint Generator with Ollama

This project generates PowerPoint presentations using Ollama's LLaMA2 model. It analyzes template structures and generates appropriate content for each slide.

## Setup

1. Install Ollama from https://ollama.ai/
2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a `templates` directory and add your PowerPoint template (template.pptx)

## Usage

```python
from powerpoint_generator import PowerPointGenerator

generator = PowerPointGenerator("templates/template.pptx")
success = generator.generate_presentation(
    "Your Presentation Topic",
    "output.pptx"
)
```

## Features

- Template structure analysis
- Content generation using Ollama's LLaMA2 model
- Support for text placeholders, tables, and images
- JSON-based content structure

## Requirements

- Python 3.8+
- Ollama
- PowerPoint template file
