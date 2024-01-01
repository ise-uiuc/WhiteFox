
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3)
        self.conv1x1 = torch.nn.Conv2d(1, 1, 1)
        self.conv3x3 = torch.nn.Conv2d(1, 1, 3, stride=2, padding=1, dilation=2, groups=1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv1x1(v1)
        v3 = self.conv3x3(v2)
        v4 = self.tanh(v3)
        v5 = torch.zeros_like(v4)
        return v5
# Inputs to the model
x = torch.randn(10, 3, 24, 24)
