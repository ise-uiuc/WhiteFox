
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(3, 100, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(100)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = self.relu
        v4 = v3(v2)
        v5 = torch.clamp(v4, 0, 6)
        v6 = v1 * v5
        v7 = v6 / 6
        v8 = self.bn(v7)
        v9 = self.tanh(v8)
        return v9
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
