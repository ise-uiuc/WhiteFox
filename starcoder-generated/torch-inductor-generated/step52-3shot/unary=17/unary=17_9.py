
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(1, 1, 3, stride=3, padding=0, dilation=1,
                                             output_padding=1, groups=1, bias=True, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(3, 1, 5, 5)
