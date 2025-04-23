#utils.py
import re
import spacy

nlp = spacy.load("en_core_web_sm")

ENTITY_PATTERNS = [
    ('email', r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    ('phone_number', r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
    ('dob', r'\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(\d{4})\b|\b\d{4}-\d{2}-\d{2}\b'),
    ('aadhar_num', r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
    ('credit_debit_no', r'\b(?:\d[ -]*?){16}\b'),
    ('cvv_no', r'\b\d{3,4}\b'),
    ('expiry_no', r'\b(0[1-9]|1[0-2])/(\d{2}|\d{4})\b')
]

def mask_pii(text: str):
    entities = []
    
    # Detect names using spaCy
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities.append({
                'start': ent.start_char,
                'end': ent.end_char,
                'type': 'full_name',
                'value': ent.text
            })
    
    # Detect other entities using regex
    for entity_type, pattern in ENTITY_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.span()
            entities.append({
                'start': start,
                'end': end,
                'type': entity_type,
                'value': match.group()
            })
    
    # Remove overlapping entities (keep longer ones)
    entities.sort(key=lambda x: x['end'] - x['start'], reverse=True)
    filtered_entities = []
    seen_spans = set()
    
    for entity in entities:
        if not any(entity['start'] >= e['start'] and entity['end'] <= e['end'] for e in filtered_entities):
            filtered_entities.append(entity)
    
    # Sort entities for replacement
    filtered_entities.sort(key=lambda x: x['start'], reverse=True)
    
    # Replace entities in text
    masked_text = text
    for entity in filtered_entities:
        masked_text = masked_text[:entity['start']] + f'[{entity["type"]}]' + masked_text[entity['end']:]
    
    return masked_text, filtered_entities
