import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentTranslator(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, out_size, kernel_size, init_identity=True):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        padding = (kernel_size - 1) // 2

        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),  # lightweight linear map
            #nn.GroupNorm(1, out_channels),  # optional normalization
        )

        #if init_identity and in_channels == out_channels and in_size == out_size:
        if init_identity:
            # Initialize Conv2d weight to identity if possible
            nn.init.dirac_(self.adapter[0].weight)
            if self.adapter[0].bias is not None:
                nn.init.zeros_(self.adapter[0].bias)

    def forward(self, z):
        # Resize spatial dimensions if needed
        if self.in_size != self.out_size:
            z = F.interpolate(z, size=self.out_size, mode='bilinear', align_corners=False)

        return self.adapter(z)



# translator = LatentTranslator(
#     in_channels=64, out_channels=64,
#     in_size=(32, 32), out_size=(16, 16), kernel_size=1,
#     init_identity=True
# )

# z_A = torch.randn(1, 64, 32, 32)  # Output from model A
# z_B = translator(z_A)             # Input for model B

# z_B_interp = F.interpolate(z_A, size=(16, 16), mode='bilinear', align_corners=False)

# print(z_B - z_B_interp)  # Should be close to zero if identity init worked