import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional
import cv2

class ImageFeatureExtractor:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        
        # Basic transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Load pre-trained model for feature extraction
        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor.fc = nn.Identity()  # Remove classification layer
        self.feature_extractor.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = self.feature_extractor.to(self.device)
    
    def extract_basic_features(self, image: Image.Image) -> Dict[str, float]:
        """Extract basic image features"""
        # Convert to numpy array
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
        
        # Quality indicators
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        noise_estimate = cv2.medianBlur(gray, 3).std()  # Estimate image noise
        
        # Professional photo indicators
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)  # Measure of object definition
        
        # Background analysis
        mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
        white_bg_ratio = np.sum(mask) / mask.size  # Ratio of white background
        
        # Color complexity
        unique_colors = len(np.unique(cv2.resize(img_bgr, (32, 32)).reshape(-1, 3), axis=0))
        color_complexity = unique_colors / (32 * 32)  # Normalized color variety
        
        return {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'sharpness': float(sharpness),
            'saturation': float(saturation),
            'noise_level': float(noise_estimate),
            'edge_density': float(edge_density),
            'white_bg_ratio': float(white_bg_ratio),
            'color_complexity': float(color_complexity)
        }
    
    def extract_deep_features(self, image: Image.Image) -> torch.Tensor:
        """Extract deep features using pre-trained model"""
        # Prepare image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(img_tensor)
        
        return features.squeeze().cpu()
    
    def process_image(self, image: Optional[Image.Image]) -> Dict[str, Any]:
        """Process image and extract all features"""
        if image is None:
            # Return zero features if image is not available
            return {
                'basic_features': {
                    'brightness': 0.0,
                    'contrast': 0.0,
                    'sharpness': 0.0,
                    'saturation': 0.0
                },
                'deep_features': torch.zeros(2048)  # ResNet50 feature size
            }
        
        # Extract all features
        basic_features = self.extract_basic_features(image)
        deep_features = self.extract_deep_features(image)
        
        return {
            'basic_features': basic_features,
            'deep_features': deep_features
        }