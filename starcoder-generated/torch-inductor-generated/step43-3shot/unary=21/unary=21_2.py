
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 1, 1, padding=2)
        self.conv2 = torch.nn.Conv2d(6, 6, 3, 1, padding=2)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(x)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x = torch.randn(2, 1, 3, 3)
# Inputs to the model end

# Model begins
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 9, 5, 1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 12, 49, 49)
