import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolution block with Conv -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    """Encoder block with two conv blocks and max pooling"""
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        pooled = self.pool(x)
        return pooled


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and two conv blocks"""
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class EncoderDecoder(nn.Module):
    """
    Simple Encoder-Decoder architecture for image segmentation
    
    Architecture:
        Encoder: 4 downsampling blocks (64->128->256->512)
        Bottleneck: 512 channels
        Decoder: 4 upsampling blocks (512->256->128->64)
        Output: 1 channel segmentation mask
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output channels (default: 1 for binary segmentation)
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(EncoderDecoder, self).__init__()
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )
        
        # Decoder
        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = DecoderBlock(64, 64)
        
        # Output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)      # -> [B, 64, H/2, W/2]
        x2 = self.enc2(x1)     # -> [B, 128, H/4, W/4]
        x3 = self.enc3(x2)     # -> [B, 256, H/8, W/8]
        x4 = self.enc4(x3)     # -> [B, 512, H/16, W/16]
        
        # Bottleneck
        x = self.bottleneck(x4)  # -> [B, 512, H/16, W/16]
        
        # Decoder
        x = self.dec4(x)       # -> [B, 256, H/8, W/8]
        x = self.dec3(x)       # -> [B, 128, H/4, W/4]
        x = self.dec2(x)       # -> [B, 64, H/2, W/2]
        x = self.dec1(x)       # -> [B, 64, H, W]
        
        # Output
        x = self.output(x)     # -> [B, 1, H, W]
        
        return x


# Test the model
if __name__ == "__main__":
    print("Testing EncoderDecoder model...")
    
    # Create model
    model = EncoderDecoder(in_channels=3, out_channels=1)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    img_size = 256
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    
    # Test with different image sizes
    print("\nTesting with different image sizes:")
    for size in [128, 256, 512]:
        x = torch.randn(2, 3, size, size)
        with torch.no_grad():
            output = model(x)
        print(f"  Input: {x.shape} -> Output: {output.shape}")
    
    print("\n✓ EncoderDecoder model test completed!")