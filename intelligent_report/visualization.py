# Visualization utilities
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Dict, Any, Optional, Union, Tuple

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')


class VisualizationManager:
    """
    Handles creation of visualizations for reports
    """
    
    def __init__(self, output_dir: str = "temp"):
        self.output_dir = output_dir
        self.visualizations = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def create_visualization(self, title: str, data: List, chart_type: str = "bar", labels: Optional[List] = None, **kwargs) -> Dict:
        """Create a visualization using raw data"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract options from kwargs
        xlabel = kwargs.pop('xlabel', None)
        ylabel = kwargs.pop('ylabel', None)
        description = kwargs.pop('description', '')
        style = kwargs.pop('style', None)
        
        # Apply style if provided
        if style:
            try:
                plt.style.use(style)
            except:
                pass  # Use default style if specified style not available
        
        # Create the specified chart type
        if chart_type.lower() == "bar":
            if labels is None:
                labels = list(range(len(data)))
            ax.bar(labels, data, **kwargs)
            plt.xticks(rotation=45, ha='right')
            
        elif chart_type.lower() == "line":
            if labels is None:
                labels = list(range(len(data)))
            ax.plot(labels, data, marker='o', **kwargs)
            plt.xticks(rotation=45, ha='right')
            
        elif chart_type.lower() == "pie":
            if labels is None:
                labels = [f"Segment {i+1}" for i in range(len(data))]
            ax.pie(data, labels=labels, autopct='%1.1f%%', shadow=True, **kwargs)
            
        elif chart_type.lower() == "scatter":
            # For scatter, data should be a list of (x,y) tuples
            x, y = zip(*data)
            ax.scatter(x, y, **kwargs)
            
        elif chart_type.lower() == "histogram":
            bins = kwargs.pop('bins', 10)
            ax.hist(data, bins=bins, **kwargs)
            
        elif chart_type.lower() == "boxplot":
            # For boxplot, data should be a list of arrays
            ax.boxplot(data, labels=labels, **kwargs)
        
        # Add labels if provided
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
            
        # Add title
        ax.set_title(title)
        plt.tight_layout()
        
        # Save the figure
        filename = f"{self.output_dir}/viz_{len(self.visualizations)}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store and return visualization metadata
        viz_data = {
            "title": title,
            "filename": filename,
            "chart_type": chart_type,
            "description": description
        }
        
        self.visualizations.append(viz_data)
        return viz_data
    
    def create_visualization_from_dataframe(self, df: pd.DataFrame, title: str, chart_type: str, 
                                           columns: List[str] = None, **kwargs) -> Dict:
        """Create visualization from a dataframe"""
        if df is None or df.empty:
            print("Error: No data provided for visualization.")
            return {}
        
        # Extract options from kwargs
        description = kwargs.pop('description', '')
        style = kwargs.pop('style', 'seaborn-v0_8-whitegrid')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Apply style
        try:
            plt.style.use(style)
        except:
            pass  # Use default style if specified style not available
            
        # Create visualization based on chart type
        if chart_type.lower() == 'bar':
            if len(columns) == 1:
                # Simple count plot
                col = columns[0]
                counts = df[col].value_counts()
                counts.plot(kind='bar', ax=ax, **{k:v for k,v in kwargs.items() if k not in ['style', 'description']})
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
            elif len(columns) == 2:
                # Grouped bar chart
                df.groupby(columns[0])[columns[1]].mean().plot(kind='bar', ax=ax)
                ax.set_ylabel(f'Mean {columns[1]}')
                
        elif chart_type.lower() == 'line':
            if 'x' in kwargs and 'y' in kwargs:
                df.plot(x=kwargs.pop('x'), y=kwargs.pop('y'), kind='line', ax=ax, marker='o')
            elif len(columns) >= 1:
                # Plot time series or sequential data
                if len(columns) == 1:
                    df[columns[0]].plot(kind='line', ax=ax, marker='o')
                else:
                    df[columns].plot(kind='line', ax=ax, marker='o')
                    
        elif chart_type.lower() == 'scatter':
            if len(columns) >= 2:
                x, y = columns[0], columns[1]
                color_col = columns[2] if len(columns) > 2 else None
                
                if color_col:
                    # Use third column for color coding
                    scatter = ax.scatter(df[x], df[y], c=df[color_col], cmap='viridis', alpha=0.7)
                    plt.colorbar(scatter, ax=ax, label=color_col)
                else:
                    ax.scatter(df[x], df[y], alpha=0.7)
                    
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                
        elif chart_type.lower() == 'histogram':
            if len(columns) >= 1:
                col = columns[0]
                ax.hist(df[col], bins=kwargs.pop('bins', 10), alpha=0.7)
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                
        elif chart_type.lower() == 'boxplot':
            if len(columns) == 1:
                # Single boxplot
                df.boxplot(column=columns[0], ax=ax)
            elif len(columns) == 2:
                # Grouped boxplot
                df.boxplot(column=columns[1], by=columns[0], ax=ax)
                plt.suptitle('')  # Remove default title
                ax.set_title(title)
                
        elif chart_type.lower() == 'heatmap':
            if len(columns) >= 2:
                # Correlation heatmap
                selected_df = df[columns] if columns else df.select_dtypes(include=[np.number])
                corr = selected_df.corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
                
        elif chart_type.lower() == 'pie':
            if len(columns) == 1:
                col = columns[0]
                counts = df[col].value_counts()
                # Limit pie chart to top categories if there are many
                if len(counts) > 8:
                    counts_limited = counts.nlargest(7)
                    counts_limited['Other'] = counts[7:].sum()
                    counts_limited.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                else:
                    counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                ax.set_ylabel('')  # Remove ylabel
        
        # Add title
        ax.set_title(title)
        plt.tight_layout()
        
        # Save the figure
        filename = f"{self.output_dir}/viz_{len(self.visualizations)}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store and return visualization metadata
        viz_data = {
            "title": title,
            "filename": filename,
            "chart_type": chart_type,
            "columns": columns,
            "description": description
        }
        
        self.visualizations.append(viz_data)
        return viz_data
