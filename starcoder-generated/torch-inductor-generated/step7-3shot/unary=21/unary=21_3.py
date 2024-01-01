
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(3, 32, 3, stride=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(32)
    def forward(self, x0):
        y4 = self.conv(x0)
        y5 = self.relu(y4)
        y6 = self.bn(y5)
        y7 = self.conv(y6)
        y8 = self.relu(y7)
        y9 = self.bn(y8)
        y10 = self.conv(y9)
        y11 = torch.tanh(y10)
        return y11
# Inputs to the model
x0 = torch.randn(1, 3, 231, 240)
