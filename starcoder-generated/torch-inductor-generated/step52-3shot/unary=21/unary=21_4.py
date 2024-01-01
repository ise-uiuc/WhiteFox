
class ModelTanh1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 1, 3, stride=2, padding=1)
        self.tanh = torch.nn.Tanh()
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self,x):
        v1 = self.conv(x)
        v2 = self.tanh(v1)
        v3 = self.tanh(v2)
        v4 = self.bn(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 64, 28, 28)
