"""PII masking module."""
import re
import logging
from typing import Dict, List, Tuple, Any

# Configure logging
logger = logging.getLogger(__name__)

# Entity patterns for regex-based detection
ENTITY_PATTERNS = [
    ('email', r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    ('phone_number', r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
    ('dob', r'\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(\d{4})\b|\b\d{4}-\d{2}-\d{2}\b'),
    ('aadhar_num', r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
    ('credit_debit_no', r'\b(?:\d[ -]*?){16}\b'),
    ('cvv_no', r'\b\d{3,4}\b'),
    ('expiry_no', r'\b(0[1-9]|1[0-2])/(\d{2}|\d{4})\b'),
    # More restrictive name detection - requires "Name is" or "My name is" prefix
    ('full_name', r'(?:(?:My name is|Name is)\s+)([A-Z][a-z]+\s+[A-Z][a-z]+)')
]

class EntityInfo:
    """Class to store entity information."""
    def __init__(self, start: int, end: int, entity_type: str, value: str):
        self.start = start
        self.end = end
        self.type = entity_type
        self.value = value
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'start': self.start,
            'end': self.end,
            'type': self.type,
            'value': self.value
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityInfo':
        """Create from dictionary."""
        return cls(
            start=data['start'],
            end=data['end'],
            entity_type=data['type'],
            value=data['value']
        )


class PIIMasker:
    """Class for masking PII in text."""
    
    def __init__(self):
        """Initialize the PII masker."""
        pass
    
    def detect_regex_entities(self, text: str) -> List[EntityInfo]:
        """Detect entities using regex patterns."""
        entities = []
        try:
            for entity_type, pattern in ENTITY_PATTERNS:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start, end = match.span()
                    
                    # For full_name pattern with capturing group, use the captured group
                    if entity_type == 'full_name' and match.groups():
                        # Adjust start position to the beginning of the capture group
                        start = match.start(1)  
                        # Get the actual name, not the prefix
                        value = match.group(1)
                    else:
                        value = match.group()
                        
                    entities.append(EntityInfo(
                        start=start,
                        end=end,
                        entity_type=entity_type,
                        value=value
                    ))
            return entities
        except Exception as e:
            logger.error(f"Error detecting regex entities: {e}")
            return []
    
    def remove_overlapping_entities(self, entities: List[EntityInfo]) -> List[EntityInfo]:
        """Remove overlapping entities, keeping the longer ones."""
        if not entities:
            return []
            
        # Sort by length (longer entities first)
        entities.sort(key=lambda x: x.end - x.start, reverse=True)
        
        filtered_entities = []
        for entity in entities:
            # Check if this entity is contained within any entity in the filtered list
            if not any(entity.start >= e.start and entity.end <= e.end for e in filtered_entities):
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def mask_text(self, text: str, entities: List[EntityInfo]) -> str:
        """Replace entities in text with their masked versions."""
        if not text or not entities:
            return text
            
        # Sort entities in reverse order (right to left) to avoid index shifting
        entities.sort(key=lambda x: x.start, reverse=True)
        
        masked_text = text
        for entity in entities:
            masked_text = masked_text[:entity.start] + f'<span class="masked-content">[{entity.type}]</span>' + masked_text[entity.end:]
        
        return masked_text
    
    def mask_pii(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Mask PII in the given text.
        
        Args:
            text: The text to mask.
            
        Returns:
            Tuple containing:
                - masked_text: Text with PII replaced by placeholders
                - entities: List of detected entities with position information
        """
        if not text:
            return text, []
        
        try:
            # Detect entities using regex only
            all_entities = self.detect_regex_entities(text)
            logger.info(f"Detected {len(all_entities)} entities")
            
            # Remove overlapping entities
            filtered_entities = self.remove_overlapping_entities(all_entities)
            logger.info(f"After removing overlaps: {len(filtered_entities)} entities")
            
            # Log what we're masking
            for entity in filtered_entities:
                logger.info(f"Masking {entity.type}: '{entity.value}' at positions {entity.start}-{entity.end}")
            
            # Mask the text
            masked_text = self.mask_text(text, filtered_entities)
            
            # Calculate % of text masked
            if text:
                original_length = len(text)
                masked_length = sum(len(f"<span class=\"masked-content\">[{e.type}]</span>") - len(e.value) for e in filtered_entities)
                mask_percentage = (masked_length / original_length) * 100
                logger.info(f"Masked approximately {mask_percentage:.1f}% of the text")
            
            # Convert entities to dictionaries
            entity_dicts = [entity.to_dict() for entity in filtered_entities]
            
            # Compare before/after
            if len(text) > 100:
                logger.info(f"Original (first 100 chars): {text[:100]}...")
                logger.info(f"Masked (first 100 chars): {masked_text[:100]}...")
            
            return masked_text, entity_dicts
            
        except Exception as e:
            logger.error(f"Error masking PII: {e}")
            return text, []


# Create a singleton instance
masker = PIIMasker()

def mask_pii(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Mask PII in the given text.
    
    This is a convenience function that uses the PIIMasker singleton.
    
    Args:
        text: The text to mask.
        
    Returns:
        Tuple containing:
            - masked_text: Text with PII replaced by placeholders
            - entities: List of detected entities with position information
    """
    return masker.mask_pii(text) 