
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 11, padding=2, stride=1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x19):
        v1 = self.conv(x19)
        v2 = torch.tanh(v1)
        v3 = self.bn(v2)
        return v3
# Inputs to the model
x19 = torch.randn(1, 3, 28, 28)
