
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 8, 1, stride=2)
    def forward(self, x1):
        v1 = F.relu(self.conv_transpose(x1))
        v2 = v1 + 0.1
        v3 = torch.clamp(v2, min=-1)
        v4 = torch.clamp(v3, max=1)
        v5 = v1 * v4
        v6 = v5 / 10
        return v6
# Inputs to the model
x1 = 200 * torch.randn(1, 1, 100, 100)
