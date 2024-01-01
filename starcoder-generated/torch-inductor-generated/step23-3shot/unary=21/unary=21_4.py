
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(4)
        self.conv = torch.nn.Conv2d(3, 4, 1, bias=None)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.bn(x1)
        v2 = self.conv(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 8, 24)
