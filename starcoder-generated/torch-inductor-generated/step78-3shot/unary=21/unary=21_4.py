
import torch.nn.functional
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 16)
        self.conv2 = torch.nn.Conv2d(16, 16, 16)
    def forward(self,x):
        v1 = self.conv1(x)
        v1 = torch.tanh(v1)
        v2 = self.conv2(v1)
        v2 = torch.tanh(v2)
        return v2
# Inputs to the model
tensor = torch.randn(1, 16, 128, 128)
