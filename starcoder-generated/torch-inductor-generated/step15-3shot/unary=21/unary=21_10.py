
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 0, stride=2, bias=False)
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.bn(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x = torch.randn(64, 1, 64, 64)
