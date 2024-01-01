
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.bn = torch.nn.BatchNorm2d(16)
        self.conv_5 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.bn(v1)
        v3 = v2 + x
        v4 = torch.relu(v3)
        v5 = self.conv_5(v4)
        v6 = v5 + v2
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
