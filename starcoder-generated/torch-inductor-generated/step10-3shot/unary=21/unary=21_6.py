
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 2, 5, stride=2, padding=1, dilation=2)
        self.norm = torch.nn.InstanceNorm2d(2, affine=True)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(5)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.norm(v1)
        v3 = self.avgpool(v2)
        return torch.tanh(v3)
# Inputs to the model
x = torch.randn(1, 16, 48, 48)
