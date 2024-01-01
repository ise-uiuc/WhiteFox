
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(48, 1024, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(1024)
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.bn(x1)
        x3 = torch.tanh(x2)
        return x3
# Inputs to the model
x = torch.randn(1, 48, 23, 23)
