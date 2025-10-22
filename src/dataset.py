import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from transformers import AutoTokenizer
import random
import numpy as np

class ProductDataset(Dataset):
    def __init__(self, csv_path, image_dir, max_text_length=64, image_size=128, is_training=True):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.is_training = is_training
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_text_length = max_text_length
        
        # Image transforms
        if is_training:
            self.transform = transforms.Compose([
                transforms.Resize((image_size + 20, image_size + 20)),
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.df)
    
    def augment_text(self, text):
        # Simple text augmentation
        words = text.split()
        if len(words) <= 3:
            return text
            
        if random.random() < 0.3:
            # Random word deletion
            num_to_delete = max(1, int(len(words) * 0.1))
            for _ in range(num_to_delete):
                if len(words) > 3:
                    del words[random.randint(0, len(words) - 1)]
        
        if random.random() < 0.3:
            # Random word shuffle
            start_idx = random.randint(0, len(words) - 3)
            end_idx = start_idx + random.randint(2, 3)
            end_idx = min(end_idx, len(words))
            section = words[start_idx:end_idx]
            random.shuffle(section)
            words[start_idx:end_idx] = section
        
        return ' '.join(words)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Process text
        text = row['catalog_content']
        if self.is_training:
            text = self.augment_text(text)
        
        # Tokenize text
        tokens = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Load and transform image
        image_path = os.path.join(self.image_dir, str(row['sample_id']) + '.jpg')
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Get price
        price = torch.tensor(row['price'], dtype=torch.float32)
        
        return {
            'text_ids': tokens['input_ids'].squeeze(0),
            'text_mask': tokens['attention_mask'].squeeze(0),
            'image': image,
            'price': price
        }