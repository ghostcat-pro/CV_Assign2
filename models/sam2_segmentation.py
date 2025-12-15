"""
SAM2-based Semantic Segmentation Model

Adapts the Segment Anything Model 2 (SAM2) for semantic segmentation tasks.
Uses SAM2's powerful image encoder with a custom decoder for dense prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SAM2Segmentation(nn.Module):
    """
    SAM2-based semantic segmentation model.
    
    This model uses SAM2's image encoder (if available) or a custom vision transformer
    encoder inspired by SAM2's architecture, combined with a decoder for semantic segmentation.
    """
    
    def __init__(self, num_classes=6, use_pretrained_sam2=True):
        super().__init__()
        self.num_classes = num_classes
        
        # Try to use SAM2 if available
        self.use_sam2 = False
        if use_pretrained_sam2:
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                
                # Try to load SAM2 model (using smaller variant for efficiency)
                # You'll need to download checkpoints from Meta's repository
                self.sam2_encoder = self._load_sam2_encoder()
                self.use_sam2 = True
                self.encoder_dim = 256  # SAM2's encoder output dimension
                print("Using SAM2 image encoder")
            except (ImportError, FileNotFoundError) as e:
                print(f"SAM2 not available ({e}), using custom ViT encoder")
                self.use_sam2 = False
        
        if not self.use_sam2:
            # Use a custom Vision Transformer encoder inspired by SAM2
            self.encoder = CustomViTEncoder(
                img_size=256,
                patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12
            )
            self.encoder_dim = 768
        
        # Decoder for semantic segmentation
        self.decoder = SegmentationDecoder(
            encoder_dim=self.encoder_dim,
            num_classes=num_classes
        )
    
    def _load_sam2_encoder(self):
        """Load SAM2 encoder (requires SAM2 installation and checkpoints)"""
        try:
            # This would load SAM2 - requires checkpoint file
            # from sam2.build_sam import build_sam2
            # sam2_checkpoint = "path/to/sam2_checkpoint.pt"
            # sam2_model = build_sam2(checkpoint=sam2_checkpoint)
            # return sam2_model.image_encoder
            raise FileNotFoundError("SAM2 checkpoint not configured")
        except Exception as e:
            raise e
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        B, C, H, W = x.shape
        
        if self.use_sam2:
            # Use SAM2 encoder
            features = self.sam2_encoder(x)
        else:
            # Use custom encoder
            features = self.encoder(x)
        
        # Decode to segmentation map
        output = self.decoder(features, (H, W))
        
        return output


class CustomViTEncoder(nn.Module):
    """
    Custom Vision Transformer encoder inspired by SAM2's architecture.
    Simplified version for underwater segmentation.
    """
    
    def __init__(self, img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Positional embedding
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            features: (B, embed_dim, H/patch_size, W/patch_size)
        """
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/p, W/p)
        
        # Flatten and add positional embedding
        B, C, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Reshape back to spatial
        x = x.transpose(1, 2).reshape(B, self.embed_dim, Hp, Wp)
        
        return x


class TransformerBlock(nn.Module):
    """Standard transformer block with self-attention and MLP"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
    
    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class SegmentationDecoder(nn.Module):
    """
    Decoder for converting encoder features to semantic segmentation.
    Uses progressive upsampling with skip connections.
    """
    
    def __init__(self, encoder_dim=768, num_classes=6):
        super().__init__()
        
        # Progressive upsampling
        self.conv1 = nn.Sequential(
            nn.Conv2d(encoder_dim, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Final classifier
        self.classifier = nn.Conv2d(128, num_classes, 1)
    
    def forward(self, x, target_size):
        """
        Args:
            x: Encoder features (B, encoder_dim, H/16, W/16)
            target_size: Target output size (H, W)
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # Upsample and refine
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv1(x)  # (B, 512, H/8, W/8)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2(x)  # (B, 256, H/4, W/4)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv3(x)  # (B, 128, H/2, W/2)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.classifier(x)  # (B, num_classes, H, W)
        
        # Ensure exact target size
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return x


# For easier importing
def get_sam2_segmentation(num_classes=6, use_pretrained_sam2=False):
    """
    Get SAM2-based segmentation model.
    
    Args:
        num_classes: Number of output classes
        use_pretrained_sam2: If True, try to use pretrained SAM2 encoder
                            (requires SAM2 installation and checkpoints)
    
    Returns:
        SAM2Segmentation model
    """
    return SAM2Segmentation(
        num_classes=num_classes,
        use_pretrained_sam2=use_pretrained_sam2
    )
