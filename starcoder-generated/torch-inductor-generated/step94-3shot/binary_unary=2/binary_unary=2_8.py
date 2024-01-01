
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 5, stride=2, padding=2)
        self.bn = torch.nn.BatchNorm2d(16)
        self.drop = torch.nn.Dropout2d(0.1)
    def forward(self, x2):
        v2 = self.conv(x2)
        v3 = v2 - 2
        v4 = F.relu(v3)
        v5 = self.bn(v4)
        v6 = v5 - 3
        v7 = F.relu(v6)
        v8 = self.drop(v7)
        return v8
# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
