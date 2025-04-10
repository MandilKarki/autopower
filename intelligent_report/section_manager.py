# Section management for reports
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import json
import os


@dataclass
class Section:
    """Represents a section in a report"""
    title: str
    content: str
    type: str = "user"  # user, ai, rag
    order: int = 0
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        """Convert section to dictionary"""
        return {
            "title": self.title,
            "content": self.content,
            "type": self.type,
            "order": self.order,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Section':
        """Create section from dictionary"""
        return cls(
            title=data.get("title", ""),
            content=data.get("content", ""),
            type=data.get("type", "user"),
            order=data.get("order", 0),
            metadata=data.get("metadata", {})
        )


class SectionManager:
    """Manages report sections"""
    
    def __init__(self):
        self.sections: Dict[str, Section] = {}
        self.section_order: List[str] = []
    
    def add_section(self, title: str, content: str, section_type: str = "user", 
                  order: Optional[int] = None, metadata: Optional[Dict] = None) -> 'SectionManager':
        """Add a section to the report"""
        # Determine order if not provided
        if order is None:
            if not self.section_order:
                order = 0
            else:
                order = max([self.sections[title].order for title in self.section_order]) + 1
        
        # Create and store section
        section = Section(
            title=title,
            content=content,
            type=section_type,
            order=order,
            metadata=metadata or {}
        )
        
        # Add to section dictionary
        self.sections[title] = section
        
        # Add to order list if not already there
        if title not in self.section_order:
            self.section_order.append(title)
        
        # Sort section order by order value
        self.section_order.sort(key=lambda x: self.sections[x].order)
        
        return self
    
    def get_section(self, title: str) -> Optional[Section]:
        """Get a section by title"""
        return self.sections.get(title)
    
    def update_section(self, title: str, content: Optional[str] = None, 
                      section_type: Optional[str] = None, order: Optional[int] = None) -> bool:
        """Update a section"""
        if title not in self.sections:
            return False
        
        section = self.sections[title]
        
        if content is not None:
            section.content = content
        
        if section_type is not None:
            section.type = section_type
        
        if order is not None:
            section.order = order
            # Re-sort section order
            self.section_order.sort(key=lambda x: self.sections[x].order)
        
        return True
    
    def remove_section(self, title: str) -> bool:
        """Remove a section"""
        if title not in self.sections:
            return False
        
        # Remove from sections dict
        del self.sections[title]
        
        # Remove from order list
        if title in self.section_order:
            self.section_order.remove(title)
        
        return True
    
    def reorder_section(self, title: str, new_position: int) -> bool:
        """Move a section to a new position"""
        if title not in self.sections or title not in self.section_order:
            return False
        
        # Remove from current position
        self.section_order.remove(title)
        
        # Insert at new position
        new_position = max(0, min(new_position, len(self.section_order)))
        self.section_order.insert(new_position, title)
        
        # Update order values for all sections
        for i, section_title in enumerate(self.section_order):
            self.sections[section_title].order = i
        
        return True
    
    def get_sections_in_order(self) -> List[Section]:
        """Get all sections in order"""
        return [self.sections[title] for title in self.section_order]
    
    def save_to_file(self, filepath: str) -> bool:
        """Save sections to a JSON file"""
        try:
            data = {
                "sections": {title: section.to_dict() for title, section in self.sections.items()},
                "order": self.section_order
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving sections: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """Load sections from a JSON file"""
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Clear current sections
            self.sections = {}
            self.section_order = []
            
            # Load sections
            for title, section_data in data.get("sections", {}).items():
                self.sections[title] = Section.from_dict(section_data)
            
            # Load order
            self.section_order = data.get("order", [])
            
            return True
        except Exception as e:
            print(f"Error loading sections: {e}")
            return False
