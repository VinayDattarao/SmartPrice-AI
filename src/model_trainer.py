import gc
import torch
import torch.nn as nn
import logging
from transformers import AutoModel
from torchvision import models

logger = logging.getLogger(__name__)

class MultiModalPricePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # Store config for later use
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = config.get('dropout', 0.1)  # Get dropout or use default
        
        # Fixed dimensions for all features
        self.text_hidden_size = 384  # DistilBERT base hidden size
        self.image_feature_size = 576  # MobileNetV3-Small feature size
        
        # Text encoder (DistilBERT) - simplified for speed
        self.text_encoder = AutoModel.from_pretrained(
            config.get('text_model', 'distilbert-base-uncased'),  # Get model name or use default
            low_cpu_mem_usage=True,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        # Set reasonable dropout
        self.text_encoder.config.dropout = 0.2
        self.text_encoder.config.attention_dropout = 0.2
        
        # Add projection layer to reduce BERT output from 768 to 384
        self.text_projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 384)
        )
        
        # MobileNetV3-Small for efficient image processing
        self.image_encoder = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.image_encoder.classifier = nn.Identity()  # Remove classifier
        
        # Text feature processors with correct dimensions for 16 features
        self.text_feature_processor = nn.Sequential(
            nn.Linear(16, 32),  # Process text statistical features
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(self.dropout),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Product info processor
        self.product_info_processor = nn.Sequential(
            nn.Linear(4, 16),  # Simple reduction
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(self.dropout),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        
        # Image feature processor
        self.image_feature_processor = nn.Sequential(
            nn.Linear(4, 16),  # Simple reduction
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(self.dropout),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        
        # Calculate exact input size for fusion
        text_features_size = 16    # From text_feature_processor
        product_info_size = 8     # From product_info_processor
        image_quality_size = 8    # From image_feature_processor
        
        fusion_input_size = (self.text_hidden_size +  # Text encoder features (384)
                           self.image_feature_size +   # Image encoder features (576)
                           text_features_size +        # Text statistical features (16)
                           product_info_size +         # Product info features (8)
                           image_quality_size)         # Image quality features (8)
        
        # Total size should be 384 + 576 + 16 + 8 + 8 = 992
        logger.info(f"Feature dimensions - BERT:{self.text_hidden_size}, "
                   f"Image:{self.image_feature_size}, Text:{text_features_size}, "
                   f"Product:{product_info_size}, Quality:{image_quality_size}, "
                   f"Total:{fusion_input_size}")
        
        # Verify dimensions match expected
        if fusion_input_size != 992:
            raise ValueError(f"Incorrect fusion input size. Expected 992, got {fusion_input_size}")
        
        # Set fusion hidden size
        fusion_hidden = 256  # Fixed reasonable size
        # More compact fusion network
        self.fusion = nn.Sequential(
            # First reduction (992 -> 256)
            nn.Linear(992, fusion_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_hidden),
            nn.Dropout(self.dropout),
            
            # Final prediction (256 -> 64 -> 1)
            nn.Linear(fusion_hidden, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1),
            nn.ReLU()  # Ensure positive prices
        )
        
        # Memory optimizations
        if torch.cuda.is_available():
            self.text_encoder.gradient_checkpointing_enable()
            if hasattr(self.text_encoder.config, 'use_memory_efficient_attention'):
                self.text_encoder.config.use_memory_efficient_attention = True
            
            # Enable activation checkpointing for EfficientNet
            for module in self.image_encoder.modules():
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = True
            
            # Convert models to float32
            self.text_encoder = self.text_encoder.float()
            self.image_encoder = self.image_encoder.float()
            
            # Move all components to device
            self.text_encoder = self.text_encoder.to(self.device)
            self.text_projection = self.text_projection.to(self.device)
            self.image_encoder = self.image_encoder.to(self.device)
            self.text_feature_processor = self.text_feature_processor.to(self.device)
            self.product_info_processor = self.product_info_processor.to(self.device)
            self.image_feature_processor = self.image_feature_processor.to(self.device)
            self.fusion = self.fusion.to(self.device)
            
            # Clear GPU cache after setup
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
        
    def calculate_loss(self, predictions, batch):
        """Calculate MSE loss for price prediction"""
        device = predictions.device
        target_prices = batch['price'].to(device, dtype=torch.float32)
        
        # MSE Loss for numerical stability
        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, target_prices)
        
        return loss
        
        # Fusion and regression layers
        fusion_input_size = (text_hidden_size +  # BERT features
                           image_feature_size +  # EfficientNet features
                           config['model']['hidden_size'] // 8 +  # Text statistical features
                           config['model']['hidden_size'] // 16 +  # Product info features
                           config['model']['hidden_size'] // 16)  # Image quality features
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, config['model']['hidden_size']),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(config['model']['hidden_size'], config['model']['hidden_size'] // 2),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(config['model']['hidden_size'] // 2, 1),
            nn.ReLU()  # Ensure positive prices
        )
        
        # Move all components to device
        self.text_encoder = self.text_encoder.to(self.device)
        self.image_encoder = self.image_encoder.to(self.device)
        self.text_feature_processor = self.text_feature_processor.to(self.device)
        self.product_info_processor = self.product_info_processor.to(self.device)
        self.image_feature_processor = self.image_feature_processor.to(self.device)
        self.fusion = self.fusion.to(self.device)
        
    def smape_loss(self, pred, target):
        """
        Differentiable SMAPE loss for direct SMAPE optimization
        """
        pred = pred.squeeze()
        target = target.squeeze()
        pred = torch.clamp(pred, min=1e-8)  # Prevent division by zero
        target = torch.clamp(target, min=1e-8)
        smape = 200 * torch.abs(pred - target) / (torch.abs(pred) + torch.abs(target))
        return torch.mean(smape)

    @torch.amp.autocast('cuda')  # Enable automatic mixed precision
    def forward(self, batch):
        # Verify batch contents
        if not all(k in batch for k in ['input_ids', 'attention_mask', 'image', 'text_features', 'product_info', 'image_features']):
            raise ValueError("Missing required batch elements")
            
        # Move batch to device efficiently
        try:
            device_batch = {
                'input_ids': batch['input_ids'].to(self.device, dtype=torch.long, non_blocking=True),
                'attention_mask': batch['attention_mask'].to(self.device, dtype=torch.long, non_blocking=True),
                'image': batch['image'].to(self.device, dtype=torch.float32, non_blocking=True),
                'text_features': batch['text_features'].to(self.device, dtype=torch.float32, non_blocking=True),
                'product_info': batch['product_info'].to(self.device, dtype=torch.float32, non_blocking=True),
                'image_features': batch['image_features'].to(self.device, dtype=torch.float32, non_blocking=True)
            }
            
            # Get batch size and verify shapes
            b = device_batch['input_ids'].size(0)  # Batch size
            max_text_length = 32  # Fixed for speed
            image_size = 64  # Fixed for speed
            assert device_batch['input_ids'].size() == (b, max_text_length), f"Input IDs shape mismatch. Expected ({b}, {max_text_length}), got {device_batch['input_ids'].size()}"
            assert device_batch['attention_mask'].size() == (b, max_text_length), f"Attention mask shape mismatch. Expected ({b}, {max_text_length}), got {device_batch['attention_mask'].size()}"
            assert device_batch['image'].size() == (b, 3, image_size, image_size), f"Image shape mismatch. Expected ({b}, 3, {image_size}, {image_size}), got {device_batch['image'].size()}"
            assert device_batch['text_features'].size() == (b, 16), f"Text features shape mismatch. Expected ({b}, 16), got {device_batch['text_features'].size()}"
            assert device_batch['product_info'].size() == (b, 4), f"Product info shape mismatch. Expected ({b}, 4), got {device_batch['product_info'].size()}"
            assert device_batch['image_features'].size() == (b, 4), f"Image features shape mismatch. Expected ({b}, 4), got {device_batch['image_features'].size()}"
            
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            raise
        
        try:
            # Process text features
            text_outputs = self.text_encoder(
                input_ids=device_batch['input_ids'],
                attention_mask=device_batch['attention_mask']
            )
            if text_outputs is None or text_outputs.last_hidden_state is None:
                raise ValueError("Text encoder returned None")
                
            # Get BERT features and project to lower dimension
            text_features = text_outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
            text_features = text_features.to(torch.float32)
            text_features = self.text_projection(text_features)  # [batch_size, 384]
            
            # Process additional features with error checking
            text_stats = self.text_feature_processor(device_batch['text_features'])
            if text_stats is None:
                raise ValueError("Text feature processing failed")
                
            product_info = self.product_info_processor(device_batch['product_info'])
            if product_info is None:
                raise ValueError("Product info processing failed")
            
            # Process image features
            image_features = self.image_encoder(device_batch['image'])
            if image_features is None:
                raise ValueError("Image encoding failed")
                
            image_quality = self.image_feature_processor(device_batch['image_features'])
            if image_quality is None:
                raise ValueError("Image quality processing failed")
                
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise
        
        try:
            # Verify all features are valid
            features = [text_features, image_features, text_stats, product_info, image_quality]
            feature_names = ['text', 'image', 'text_stats', 'product_info', 'image_quality']
            
            # Log shapes for debugging
            for i, (f, name) in enumerate(zip(features, feature_names)):
                if f is None:
                    raise ValueError(f"{name} features are None")
                if torch.isnan(f).any():
                    raise ValueError(f"Found NaN values in {name} features")
                logger.info(f"{name} features shape: {f.shape}")
            
            # Memory-efficient feature combination
            combined_features = torch.cat([
                text_features,      # Semantic features (384)
                image_features,     # MobileNet features (576)
                text_stats,         # Statistical text features (16)
                product_info,       # Product info (8)
                image_quality       # Image quality metrics (8)
            ], dim=1)  # Total: 992
            
            # Verify combined features shape
            expected_size = 992  # 384 + 576 + 16 + 8 + 8
            if combined_features.size(1) != expected_size:
                raise ValueError(f"Combined features dimension mismatch. Expected {expected_size}, got {combined_features.size(1)}")
            
            # Price prediction
            price_pred = self.fusion(combined_features)
            if price_pred is None:
                raise ValueError("Fusion layer returned None")
            
            price_pred = price_pred.squeeze()
            
            # Verify prediction
            if torch.isnan(price_pred).any():
                raise ValueError("NaN values in predictions")
            if not torch.all(price_pred >= 0):
                raise ValueError("Negative price predictions found")
            
            # Log prediction stats for debugging
            logger.info(f"Price predictions - Mean: {price_pred.mean():.2f}, Min: {price_pred.min():.2f}, Max: {price_pred.max():.2f}")
            
            # Clean up intermediate tensors
            del text_features, image_features, text_stats, product_info, image_quality, combined_features
            torch.cuda.empty_cache()
            
            return price_pred
            
        except Exception as e:
            logger.error(f"Error in feature combination: {str(e)}")
            torch.cuda.empty_cache()  # Clean up in case of error
            raise