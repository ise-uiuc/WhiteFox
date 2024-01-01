
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(6, 6, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
tensor = torch.randn(1, 1, 64, 64)
#Model ends

# Model begins
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        a = self.conv(x)
        b = self.tanh(a)
        return b
# Inputs to the model
tensor = torch.randn(1,1,64,64)
