import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 128, 4, 2, 1, bias=False),  # [B, 3, 64, 64] -> [B, 128, 32, 32]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),       # [B, 128, 32, 32] -> [B, 256, 16, 16]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),       # [B, 256, 16, 16] -> [B, 512, 8, 8]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),         # [B, 512, 8, 8] -> [B, 1, 1, 1]
            nn.Sigmoid()
        )

    def forward(self, img):
        # Forward pass through the Discriminator
        output = self.model(img)  # [B, 1, 1, 1]
        return output.view(img.size(0), -1)  # Flatten to [B, 1]

# For testing purposes
if __name__ == "__main__":
    import torch
    channels = 3  # Example for RGB images
    discriminator = Discriminator(channels=channels)
    print(discriminator)

    # Dummy input tensor
    img = torch.randn(8, channels, 64, 64)  # Batch of 8 RGB images of size 64x64
    output = discriminator(img)
    print(f"Input shape: {img.shape}, Output shape: {output.shape}")
