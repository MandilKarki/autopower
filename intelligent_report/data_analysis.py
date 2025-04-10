# Data analysis utilities
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple


class DataAnalyzer:
    """
    Handles data loading and analysis for reporting
    """
    
    def __init__(self):
        self.dataset = None
        self.data_summary = None
    
    def load_data(self, file_path: str) -> bool:
        """Load data from CSV, Excel, or JSON file"""
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            return False
            
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                self.dataset = pd.read_csv(file_path)
            elif file_ext in ['.xls', '.xlsx']:
                self.dataset = pd.read_excel(file_path)
            elif file_ext == '.json':
                self.dataset = pd.read_json(file_path)
            else:
                print(f"Error: Unsupported file format {file_ext}")
                return False
                
            # Successfully loaded data
            print(f"Data loaded successfully: {self.dataset.shape[0]} rows, {self.dataset.shape[1]} columns")
            
            # Analyze dataset
            self._analyze_dataset()
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _analyze_dataset(self) -> None:
        """Analyze dataset and store insights"""
        if self.dataset is None or self.dataset.empty:
            self.data_summary = None
            return
            
        # Store dataset summary
        self.data_summary = {
            "rows": self.dataset.shape[0],
            "columns": self.dataset.shape[1],
            "column_types": {col: str(dtype) for col, dtype in self.dataset.dtypes.items()},
            "missing_values": self.dataset.isnull().sum().to_dict(),
            "numeric_columns": self.dataset.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": self.dataset.select_dtypes(include=['object']).columns.tolist(),
        }
        
        # Calculate summary statistics for numeric columns
        if self.data_summary["numeric_columns"]:
            self.data_summary["numeric_stats"] = self.dataset[self.data_summary["numeric_columns"]].describe().to_dict()
        
        # Calculate value counts for categorical columns (limited to top 10)
        if self.data_summary["categorical_columns"]:
            self.data_summary["categorical_stats"] = {}
            for col in self.data_summary["categorical_columns"]:
                self.data_summary["categorical_stats"][col] = self.dataset[col].value_counts().head(10).to_dict()
    
    def suggest_visualizations(self) -> List[Dict]:
        """Suggest visualizations based on dataset structure"""
        if self.dataset is None or self.data_summary is None:
            return []
            
        suggestions = []
        
        numeric_cols = self.data_summary["numeric_columns"]
        categorical_cols = self.data_summary["categorical_columns"]
        
        # Suggestion 1: If we have numeric columns, suggest distributions
        if numeric_cols:
            for col in numeric_cols[:3]:  # Limit to first 3 to avoid too many suggestions
                suggestions.append({
                    "title": f"Distribution of {col}",
                    "type": "histogram",
                    "columns": [col],
                    "description": f"Histogram showing the distribution of {col} values."
                })
                
        # Suggestion 2: If we have categorical columns, suggest count plots
        if categorical_cols:
            for col in categorical_cols[:3]:  # Limit to first 3
                suggestions.append({
                    "title": f"Count of {col} Categories",
                    "type": "bar",
                    "columns": [col],
                    "description": f"Bar chart showing the count of each {col} category."
                })
                
        # Suggestion 3: If we have both numeric and categorical, suggest relationship
        if numeric_cols and categorical_cols:
            suggestions.append({
                "title": f"Relationship: {categorical_cols[0]} vs {numeric_cols[0]}",
                "type": "boxplot",
                "columns": [categorical_cols[0], numeric_cols[0]],
                "description": f"Box plot showing {numeric_cols[0]} distribution across {categorical_cols[0]} categories."
            })
            
        # Suggestion 4: If we have multiple numeric columns, suggest correlation
        if len(numeric_cols) >= 2:
            suggestions.append({
                "title": "Correlation Matrix",
                "type": "heatmap",
                "columns": numeric_cols[:5],  # Limit to 5 columns for readability
                "description": "Heatmap showing correlations between numeric variables."
            })
            
            suggestions.append({
                "title": f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}",
                "type": "scatter",
                "columns": [numeric_cols[0], numeric_cols[1]],
                "description": f"Scatter plot showing relationship between {numeric_cols[0]} and {numeric_cols[1]}."
            })
            
        return suggestions
        
    def get_column_data(self, column_name: str) -> Tuple[List, List]:
        """Get data and labels from a specified column"""
        if self.dataset is None or column_name not in self.dataset.columns:
            return [], []
        
        if column_name in self.data_summary["categorical_columns"]:
            # For categorical, return value counts
            counts = self.dataset[column_name].value_counts()
            return counts.values.tolist(), counts.index.tolist()
        else:
            # For numeric, return the values directly
            return self.dataset[column_name].tolist(), []
            
    def get_grouped_data(self, category_col: str, value_col: str, aggregation: str = 'mean') -> Tuple[List, List]:
        """Get aggregated data grouped by a category"""
        if self.dataset is None or category_col not in self.dataset.columns or value_col not in self.dataset.columns:
            return [], []
            
        try:
            if aggregation == 'mean':
                grouped = self.dataset.groupby(category_col)[value_col].mean()
            elif aggregation == 'sum':
                grouped = self.dataset.groupby(category_col)[value_col].sum()
            elif aggregation == 'count':
                grouped = self.dataset.groupby(category_col)[value_col].count()
            else:
                grouped = self.dataset.groupby(category_col)[value_col].mean()  # Default to mean
                
            return grouped.values.tolist(), grouped.index.tolist()
        except Exception as e:
            print(f"Error getting grouped data: {e}")
            return [], []
