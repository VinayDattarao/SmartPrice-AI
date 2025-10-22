import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from .text_processor import preprocess_catalog_content

def extract_image_features(image):
    """Extract basic image features"""
    if image is None:
        return {
            'brightness': 0.0,
            'contrast': 0.0,
            'sharpness': 0.0,
            'saturation': 0.0
        }
        
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to BGR for OpenCV
    if len(img_array.shape) == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    
    # Calculate basic statistics
    brightness = np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0])
    contrast = np.std(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0])
    sharpness = cv2.Laplacian(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    
    # Calculate color features
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1])
    
    return {
        'brightness': float(brightness),
        'contrast': float(contrast),
        'sharpness': float(sharpness),
        'saturation': float(saturation)
    }

class ProductDataset(Dataset):
    def __init__(self, data, image_dir, tokenizer, max_length=32, image_size=64, transform=None):
        print(f"Loading CSV data")
        self.data = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)
        print(f"Loaded {len(self.data)} samples")
        
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        print("Using provided tokenizer")
        
        self.max_length = max_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Preprocess text data
        print("Starting text preprocessing...")
        self.processed_texts = {}
        from tqdm import tqdm
        for idx, row in tqdm(enumerate(self.data.iterrows()), total=len(self.data), desc="Processing texts"):
            self.processed_texts[idx] = preprocess_catalog_content(row[1]['catalog_content'])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        processed_text = self.processed_texts[idx]
        
        # Process text - squeeze to remove batch dimension
        text_encoding = self.tokenizer(
            processed_text['cleaned_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        text_encoding = {k: v.squeeze(0) for k, v in text_encoding.items()}  # Remove batch dimension
        
        # Get text features (16 features)
        text_features = torch.tensor([
            # Core stats (4)
            processed_text['features']['text_length'],
            processed_text['features']['word_count'],
            processed_text['features']['avg_word_length'],
            processed_text['features']['unique_words'],
            # Text composition (4)
            processed_text['features']['text_length'] / 1000.0,  # Normalized length
            processed_text['features']['word_count'] / 100.0,  # Normalized word count
            processed_text['features']['avg_word_length'] / 10.0,  # Normalized avg length
            len(set(processed_text['features'])) / 20.0,  # Feature variety
            # Sentence stats (4)
            processed_text['features'].get('sentences', 0.0),
            processed_text['features'].get('avg_sentence_length', 0.0),
            processed_text['features'].get('sentences', 0.0) / 10.0,  # Normalized sentence count
            processed_text['features'].get('avg_sentence_length', 0.0) / 20.0,  # Normalized avg sentence length
            # Character stats (4)
            processed_text['features'].get('num_digits', 0.0),
            processed_text['features'].get('num_special_chars', 0.0),
            processed_text['features'].get('num_digits', 0.0) / (processed_text['features']['text_length'] + 1e-8),
            processed_text['features'].get('num_special_chars', 0.0) / (processed_text['features']['text_length'] + 1e-8)
        ], dtype=torch.float32)
        
        # Get product info features
        product_info = torch.tensor([
            float(processed_text['features']['ipq']),
            processed_text['features']['value'],
            float(processed_text['features']['has_value_unit']),
            1.0 if processed_text['features']['unit'] == 'Count' else 0.0
        ], dtype=torch.float32)
        
        # Load and process image
        image_path = f"{self.image_dir}/{row['sample_id']}.jpg"
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            # Extract image features
            image_features = extract_image_features(image)
        except:
            # Return blank image if file not found
            image_tensor = torch.zeros(3, self.transform.transforms[0].size[0], 
                                    self.transform.transforms[0].size[1])
            image_features = extract_image_features(None)
        
        # Create image features tensor
        image_features_tensor = torch.tensor([
            image_features['brightness'],
            image_features['contrast'],
            image_features['sharpness'],
            image_features['saturation']
        ], dtype=torch.float32)
        
        # Get price (if available)
        price = torch.tensor(row['price'], dtype=torch.float32) if 'price' in row else torch.tensor(0.0, dtype=torch.float32)
        price = price.to(torch.float32)  # Ensure price is float32
        
        return {
            'input_ids': text_encoding['input_ids'].squeeze(),
            'attention_mask': text_encoding['attention_mask'].squeeze(),
            'text_features': text_features,
            'product_info': product_info,
            'image': image_tensor,
            'image_features': image_features_tensor,
            'price': price,
            'sample_id': row['sample_id']
        }