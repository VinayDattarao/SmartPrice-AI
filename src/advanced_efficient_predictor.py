import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, context):
        b, n, d = x.shape
        h = self.num_heads
        
        q = self.to_q(x).view(b, n, h, -1).transpose(1, 2)
        k = self.to_k(context).view(b, n, h, -1).transpose(1, 2)
        v = self.to_v(context).view(b, n, h, -1).transpose(1, 2)
        
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.to_out(out)
        return self.norm(out + x)  # Add residual connection

class PricePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Model dimensions
        self.hidden_dim = config.get('hidden_size', 512)
        self.num_heads = config.get('attention_heads', 8)
        
        # Text encoder (DistilBERT)
        self.text_encoder = AutoModel.from_pretrained(config['text_model'])
        self.text_dim = self.text_encoder.config.hidden_size
        
        # Image encoder (EfficientNet-B0)
        self.image_encoder = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-1])
        self.image_dim = 1280  # EfficientNet-B0 feature dimension
        
        # Feature projections with LayerNorm
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1))
        )
        
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1))
        )
        
        # Cross attention layers
        self.text_to_image = CrossAttention(self.hidden_dim, self.num_heads)
        self.image_to_text = CrossAttention(self.hidden_dim, self.num_heads)
        
        # Price range embedding
        self.price_range_embedding = nn.Parameter(torch.randn(10, self.hidden_dim))
        
        # Final layers
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1))
        )
        
        self.price_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
    def forward(self, text_ids, text_mask, images):
        # Text encoding
        text_output = self.text_encoder(input_ids=text_ids, attention_mask=text_mask)
        text_features = text_output.last_hidden_state[:, 0]  # Use [CLS] token
        text_features = self.text_proj(text_features)
        
        # Image encoding
        image_features = self.image_encoder(images)
        image_features = image_features.squeeze(-1).squeeze(-1)
        image_features = self.image_proj(image_features)
        
        # Cross attention
        text_attended = self.text_to_image(
            text_features.unsqueeze(1), 
            image_features.unsqueeze(1)
        ).squeeze(1)
        
        image_attended = self.image_to_text(
            image_features.unsqueeze(1), 
            text_features.unsqueeze(1)
        ).squeeze(1)
        
        # Add learned price range embeddings
        price_context = self.price_range_embedding.mean(0).expand(text_features.size(0), -1)
        
        # Combine all features
        combined = torch.cat([
            text_attended,
            image_attended,
            price_context
        ], dim=-1)
        
        # Final prediction
        fused = self.fusion(combined)
        price = self.price_head(fused).squeeze(-1)
        
        return price

    def predict(self, text_ids, text_mask, images):
        with torch.no_grad():
            return self.forward(text_ids, text_mask, images)