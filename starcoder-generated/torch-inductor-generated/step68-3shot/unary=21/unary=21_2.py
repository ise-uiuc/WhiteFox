
import torch.nn
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 64, 12, stride=12, padding=6)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        t1 = self.conv(x)
        t2 = self.tanh(t1)
        return t2
# Inputs to the model
x = torch.randn(1, 1, 64, 64)
