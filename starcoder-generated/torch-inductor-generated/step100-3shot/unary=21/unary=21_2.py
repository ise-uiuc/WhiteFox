
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 1)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.bn(v1)
        return torch.tanh(v2)
# Inputs to the model
x = torch.randn(1, 1, 32, 32)
