import re
import html
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """
    Clean and preprocess text data
    """
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Quick cleaning for fast processing
    text = re.sub(r'<[^>]+>|http\S+|www\S+|https\S+|[^\w\s]|\d+', ' ', text)
    
    # Remove multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_text_features(text: str) -> Dict[str, Any]:
    """
    Extract basic text features
    """
    # Tokenize with error handling
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()  # Fallback to simple splitting if NLTK fails
    
    # Get stopwords with error handling
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        stop_words = set(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it',
                         'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'])
    
    # Define IPQ patterns for price indication
    ipq_patterns = {
        'premium': r'premium|luxury|high[- ]end|exclusive|designer|limited|special',
        'basic': r'basic|standard|regular|normal|common|simple',
        'quality': r'quality|excellent|superior|best|finest|top[- ]grade',
        'material': r'leather|cotton|silk|wool|metal|steel|plastic|glass|wooden|wood',
        'brand': r'brand|original|authentic|genuine|official',
        'condition': r'new|used|refurbished|unopened|sealed'
    }
    
    # Extract IPQ features
    ipq_features = {
        f'has_{key}': 1 if re.search(pattern, text, re.I) else 0
        for key, pattern in ipq_patterns.items()
    }
    
    # Remove stopwords and empty tokens
    tokens = [t for t in tokens if t and t not in stop_words]
    
    features = {
        # Basic text features
        'text_length': len(text),
        'word_count': len(tokens),
        'avg_word_length': sum(len(t) for t in tokens) / len(tokens) if tokens else 0,
        'unique_words': len(set(tokens)),
        # Additional features
        'sentences': len(re.split(r'[.!?]+', text)),  # Number of sentences
        'avg_sentence_length': len(tokens) / len(re.split(r'[.!?]+', text)) if text else 0,  # Avg words per sentence
        'num_digits': len(re.findall(r'\d+', text)),  # Number of digits
        'num_special_chars': len(re.findall(r'[^a-zA-Z0-9\s]', text)),  # Number of special characters
        **ipq_features  # Add IPQ features
    }
    
    return features

def extract_product_info(text: str) -> dict:
    """
    Extract product information including IPQ, value, and unit
    """
    info = {
        'ipq': 1,
        'value': 1.0,
        'unit': 'Count',
        'has_value_unit': False
    }
    
    # Extract Value and Unit if present
    value_pattern = r'Value:\s*([\d.]+)'
    unit_pattern = r'Unit:\s*(\w+)'
    
    value_match = re.search(value_pattern, text)
    unit_match = re.search(unit_pattern, text)
    
    if value_match and unit_match:
        info['value'] = float(value_match.group(1))
        info['unit'] = unit_match.group(1)
        info['has_value_unit'] = True
    
    # Common IPQ patterns
    ipq_patterns = [
        r'pack of (\d+)',
        r'(\d+)[\s-]pack',
        r'(\d+)[\s-]count',
        r'(\d+)[\s-]piece',
        r'(\d+)[\s-]pc',
        r'quantity:?\s*(\d+)',
        r'qty:?\s*(\d+)',
        r'set of (\d+)',
        r'(\d+)[\s-]set'
    ]
    
    # Try to find IPQ in text
    for pattern in ipq_patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                info['ipq'] = int(match.group(1))
                break
            except ValueError:
                continue
    
    return info

def preprocess_catalog_content(text: str) -> Dict[str, Any]:
    """
    Process catalog content and extract all relevant features
    """
    # Split into sections if possible
    sections = {
        'title': '',
        'bullet_points': [],
        'description': '',
        'metadata': ''
    }
    
    lines = text.split('\n')
    current_section = 'title'
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Item Name:'):
            sections['title'] = line.replace('Item Name:', '').strip()
        elif line.startswith('Bullet Point'):
            point = line.split(':', 1)[1].strip() if ':' in line else line
            sections['bullet_points'].append(point)
        elif line.startswith('Product Description:'):
            sections['description'] = line.replace('Product Description:', '').strip()
        elif line.startswith(('Value:', 'Unit:')):
            sections['metadata'] += line + '\n'
    
    # Clean text for each section
    cleaned_title = clean_text(sections['title'])
    cleaned_bullets = [clean_text(bp) for bp in sections['bullet_points']]
    cleaned_desc = clean_text(sections['description'])
    
    # Combine cleaned text
    cleaned_text = cleaned_title + ' ' + ' '.join(cleaned_bullets) + ' ' + cleaned_desc
    
    # Extract basic text features
    features = extract_text_features(cleaned_text)
    
    # Extract product info (IPQ, value, unit)
    product_info = extract_product_info(text)
    features.update(product_info)
    
    # Add section-specific features
    features.update({
        'title_length': len(cleaned_title),
        'num_bullet_points': len(sections['bullet_points']),
        'description_length': len(cleaned_desc),
        'has_structured_format': bool(sections['title'] and sections['bullet_points'])
    })
    
    return {
        'cleaned_text': cleaned_text,
        'features': features,
        'sections': sections
    }