
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 7, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(7)
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.bn(v1)
        v3 = self.relu(v2)
        v4 = v3 > 0
        v5 = v3 * 0.2
        v6 = torch.where(v4, v3, v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 7, 31, 31)
