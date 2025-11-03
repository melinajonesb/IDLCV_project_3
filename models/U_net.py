import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Double convolution block: (Conv -> BatchNorm -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block: MaxPool -> DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block: Upsample -> Concat -> DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        # Use bilinear upsampling or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Upsampled feature map from decoder
            x2: Skip connection from encoder
        """
        x1 = self.up(x1)
        
        # Handle size mismatch between x1 and x2
        # Input shape: [B, C, H, W]
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for image segmentation
    
    Original paper: https://arxiv.org/abs/1505.04597
    
    Architecture:
        Encoder (Contracting Path):
            - 4 downsampling blocks with doubling channels (64->128->256->512)
            - Each block: Conv->BN->ReLU->Conv->BN->ReLU->MaxPool
        
        Bottleneck:
            - DoubleConv with 1024 channels
        
        Decoder (Expanding Path):
            - 4 upsampling blocks with halving channels (512->256->128->64)
            - Each block: Upsample->Concat with skip connection->DoubleConv
        
        Output:
            - 1x1 Conv to produce segmentation mask
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output channels (default: 1 for binary segmentation)
        bilinear: Use bilinear upsampling instead of transposed convolution (default: True)
    """
    def __init__(self, in_channels=3, out_channels=1, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Initial convolution (no downsampling)
        self.inc = DoubleConv(in_channels, 64)
        
        # Encoder (downsampling path)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # Adjust factor based on upsampling method
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder (upsampling path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output convolution
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)       # -> [B, 64, H, W]
        x2 = self.down1(x1)    # -> [B, 128, H/2, W/2]
        x3 = self.down2(x2)    # -> [B, 256, H/4, W/4]
        x4 = self.down3(x3)    # -> [B, 512, H/8, W/8]
        x5 = self.down4(x4)    # -> [B, 512 or 1024, H/16, W/16]
        
        # Decoder with skip connections
        x = self.up1(x5, x4)   # -> [B, 256 or 512, H/8, W/8]
        x = self.up2(x, x3)    # -> [B, 128 or 256, H/4, W/4]
        x = self.up3(x, x2)    # -> [B, 64 or 128, H/2, W/2]
        x = self.up4(x, x1)    # -> [B, 64, H, W]
        
        # Output
        logits = self.outc(x)  # -> [B, 1, H, W]
        
        return logits


# Test the model
if __name__ == "__main__":
    print("Testing U-Net model...")
    
    # Create model
    model = UNet(in_channels=3, out_channels=1, bilinear=True)
    
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
    
    # Verify output dimensions match input
    assert output.shape == (batch_size, 1, img_size, img_size), "Output shape mismatch!"
    print("✓ Output dimensions correct")
    
    # Test with different image sizes
    print("\nTesting with different image sizes:")
    for size in [128, 256, 512]:
        x = torch.randn(2, 3, size, size)
        with torch.no_grad():
            output = model(x)
        print(f"  Input: {x.shape} -> Output: {output.shape}")
        assert output.shape == (2, 1, size, size), f"Size mismatch for {size}x{size}!"
    
    # Compare bilinear vs transposed convolution
    print("\nComparing upsampling methods:")
    model_bilinear = UNet(in_channels=3, out_channels=1, bilinear=True)
    model_transposed = UNet(in_channels=3, out_channels=1, bilinear=False)
    
    params_bilinear = sum(p.numel() for p in model_bilinear.parameters())
    params_transposed = sum(p.numel() for p in model_transposed.parameters())
    
    print(f"  Bilinear upsampling: {params_bilinear:,} parameters")
    print(f"  Transposed conv: {params_transposed:,} parameters")
    
    print("\n✓ U-Net model test completed!")