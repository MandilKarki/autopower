# Example usage of the Intelligent Report Generator
import os
import random
import json
import pandas as pd
from intelligent_report.intelligent_report import IntelligentReportGenerator


def create_template():
    """Create a sample template config file"""
    template = {
        "report": {
            "title": "Market Analysis Report Template",
            "subtitle": "Comprehensive Market Insights",
            "author": "AI Report Generator",
            "company": "Autopower Analytics"
        },
        "sections": {
            "executive_summary": {
                "title": "Executive Summary",
                "type": "user",
                "order": 1,
                "template": "This section should provide a concise overview of the report findings.\n\nKey points to address:\n- Market overview\n- Major trends identified\n- Critical insights for decision makers\n- Recommendations"
            },
            "industry_overview": {
                "title": "Industry Overview",
                "type": "ai",
                "order": 2,
                "prompt": "Current state of the technology industry, including major trends, disruptions, and growth sectors",
                "detailed": True
            },
            "key_trends": {
                "title": "Key Market Trends",
                "type": "ai",
                "order": 3,
                "prompt": "Emerging trends in market consumer behavior and technology adoption patterns",
                "detailed": False
            },
            "competitive_landscape": {
                "title": "Competitive Landscape",
                "type": "user",
                "order": 4,
                "template": "Analysis of major competitors and market positioning.\n\nThis section should be customized with specific company information."
            }
        },
        "visualizations": {
            "auto_generate": True,
            "auto_count": 2,
            "custom": [
                {
                    "title": "Sector Growth Comparison",
                    "type": "bar",
                    "columns": ["Sector", "Growth"],
                    "description": "Comparison of growth rates across different market sectors"
                },
                {
                    "title": "Revenue Distribution",
                    "type": "boxplot",
                    "columns": ["Sector", "Revenue"],
                    "description": "Distribution of revenue across market sectors"
                }
            ]
        },
        "output": {
            "markdown": "output/report.md",
            "powerpoint": "output/presentation.pptx",
            "visualization_dir": "output/images"
        }
    }
    
    # Create templates directory if it doesn't exist
    os.makedirs("templates", exist_ok=True)
    
    # Save the template to a file
    with open("templates/market_analysis_template.json", "w") as f:
        json.dump(template, f, indent=4)
    
    return "templates/market_analysis_template.json"


def main():
    print("\nIntelligent Report Generator Demo\n" + "=" * 30)
    
    # Create sample directories
    os.makedirs("temp", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/images", exist_ok=True)
    
    # Create a sample template file
    template_path = create_template()
    print(f"Created template: {template_path}")
    
    # 1. Test creating report from a template
    print("\n1. Creating a report from template:\n==============================")
    report = IntelligentReportGenerator(config_path=template_path)
    
    # Create sample data
    print("Creating sample dataset...")
    sample_data = {
        "Sector": ["Tech", "Healthcare", "Finance", "Retail", "Energy"] * 5,
        "Revenue": [random.randint(50, 200) for _ in range(25)],
        "Growth": [random.uniform(-5, 25) for _ in range(25)],
        "Employees": [random.randint(100, 5000) for _ in range(25)],
        "Satisfaction": [random.uniform(3.0, 5.0) for _ in range(25)]
    }
    df = pd.DataFrame(sample_data)
    df.to_csv("temp/sample_data.csv", index=False)
    
    # Load the data for analysis and visualization
    report.load_data_from_file("temp/sample_data.csv")
    
    # Add a custom section that's not in the template
    report.add_user_section(
        "Future Outlook",
        "This section covers projections and forecasts for the next 1-3 years based on current trends.\n"
        "Market experts predict continued growth in the technology and healthcare sectors, with\n"
        "potential disruption in traditional retail and finance models due to technology adoption.",
        order=5  # Place it after the competitive landscape section
    )
    
    # Generate reports from template
    print("Generating report and presentation from template...")
    md_path = report.generate_report()
    ppt_path = report.generate_presentation()
    
    print(f"Generated report from template: {md_path}")
    print(f"Generated presentation from template: {ppt_path}")
    
    # 2. Test creating a custom report without template
    print("\n2. Creating a custom report without template:\n======================================")
    custom_report = IntelligentReportGenerator("Custom Market Analysis 2024")
    
    # Load the same dataset
    custom_report.load_data_from_file("temp/sample_data.csv")
    
    # Get topic suggestions based on a general area
    print("\nSuggested report topics:\n=======================")
    suggested_topics = custom_report.suggest_topics("market analysis", count=3)
    for i, topic in enumerate(suggested_topics, 1):
        print(f"{i}. {topic['title']}: {topic['description']}")
    
    # Add sections with specific ordering
    custom_report.add_user_section(
        "Executive Summary",
        "This report provides a comprehensive analysis of market trends across key sectors in 2024.\n"
        "The findings presented here are based on data collected from multiple sources and analyzed using\n"
        "advanced statistical methods and AI techniques.",
        order=1
    )
    
    custom_report.generate_ai_section(
        "Industry Overview",
        "Current state of the technology industry in 2024, including major trends, disruptions, and growth sectors.",
        detailed=True,
        order=2
    )
    
    # Add custom visualization
    market_data = [42, 35, 63, 27, 45]
    market_labels = ["Tech", "Healthcare", "Finance", "Retail", "Energy"]
    custom_report.add_visualization(
        "Market Growth Forecast",
        market_data,
        "bar",
        market_labels,
        xlabel="Industry Sector",
        ylabel="Projected Growth (%)",
        description="Projected growth percentages across major industry sectors for 2024-2025."
    )
    
    # Generate the reports
    custom_md_path = custom_report.generate_report("output/custom_report.md")
    custom_ppt_path = custom_report.generate_presentation("output/custom_presentation.pptx")
    
    # Save this as a template for future use
    custom_template_path = "templates/custom_template.json"
    custom_report.save_template(custom_template_path)
    
    print(f"Generated custom report: {custom_md_path}")
    print(f"Generated custom presentation: {custom_ppt_path}")
    print(f"Saved custom template: {custom_template_path}")
    
    print("\nAll files generated successfully:\n============================")
    print(f"1. Template-based report: {md_path}")
    print(f"2. Template-based presentation: {ppt_path}")
    print(f"3. Custom report: {custom_md_path}")
    print(f"4. Custom presentation: {custom_ppt_path}")
    print(f"5. Sample template: {template_path}")
    print(f"6. Custom template: {custom_template_path}")



if __name__ == "__main__":
    main()
