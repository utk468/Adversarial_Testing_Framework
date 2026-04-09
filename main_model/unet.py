import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),

                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.enc1 = block(3, 64)
        self.enc2 = block(64, 128)
        self.enc3 = block(128, 256)

        self.pool = nn.MaxPool2d(2, 2)

        # Decoder (with skip connections)
        self.dec3 = block(256 + 128, 128)
        self.dec2 = block(128 + 64, 64)

        # Output logits
        self.out = nn.Conv2d(64, 1, kernel_size=1)

        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)               # [B, 64, H, W]
        x2 = self.enc2(self.pool(x1))   # [B, 128, H/2, W/2]
        x3 = self.enc3(self.pool(x2))   # [B, 256, H/4, W/4]

        # Decoder
        x = self.up(x3)                 # [B, 256, H/2, W/2]
        x = torch.cat([x, x2], dim=1)
        x = self.dec3(x)

        x = self.up(x)                  # [B, 128, H, W]
        x = torch.cat([x, x1], dim=1)
        x = self.dec2(x)

        return self.out(x)              # LOGITS

