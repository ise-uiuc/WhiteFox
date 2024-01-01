
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 4, (1, 1))
        self.pad = torch.nn.ReflectionPad2d(1)
    def forward(self, x):
        v1 = self.pad(x)
        v2 = self.conv(v1)
        v3 = torch.tanh(v2)
        return v3
# Input for the model
x = torch.randn(1, 6, 64, 64)
# model end

# Model begins
import torch
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        o1 = torch.tanh(x)
        o2 = torch.tanh(x.permute(0, 2, 3, 1))
        o3 = torch.tanh(torch.mul(x, x))
        return o3
# Input for the model
x = torch.randn(1, 3, 4, 5)
# Model end

# Model begins
import torch
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v2 = torch.tanh(x)
        v3 = torch.tanh(x.view(x.size()[2:-1]))
        v4 = torch.tanh(x.view(x.size()[-2:]))
        return v4
# Input for the model
x = torch.randn(1, 4, 5)
