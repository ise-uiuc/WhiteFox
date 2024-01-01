
import torch
torch.manual_seed(1)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(in_channels=20, out_channels=128, kernel_size=3, activation='relu')
    def forward(self, x1):
        # PyTorch does not support negative stride or padding, so I use a new trick as follows
        # Apply convolution operation to the reversed input tensor
        x1 = x1.flip(1).transpose(0, 1) # Reversed channels
        v1 = self.conv(x1) # Convolution operation to the reversed input tensor
        # Recover the original shape
        v2 = v1.transpose(0, 1).flip(1) # Reversed channels
        return v2
# Inputs to the model
x1 = torch.randn(20, 30)
