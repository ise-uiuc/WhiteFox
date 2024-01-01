
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv1x1 = torch.nn.Conv2d(16, 8, 1)
        self.conv3x3 = torch.nn.Conv2d(8, 8, 3, dilation=2, padding=3)
        self.conv = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.conv(self.conv(self.conv(x)))
        v2 = self.conv1x1(v1)
        v3 = torch.tanh(self.conv(self.conv3x3(v2)))
        return v3
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
