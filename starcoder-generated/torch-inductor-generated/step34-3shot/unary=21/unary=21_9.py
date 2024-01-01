
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 12, 8, stride=3, padding=2)
        self.conv2 = torch.nn.Conv2d(12, 24, 12, 15, padding=4, dilation=3)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
tensor = torch.randn(64, 1, 224, 224)
