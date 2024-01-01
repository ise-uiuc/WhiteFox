
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(32)
        self.globalpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(32, 10)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = torch.clamp(v2, min=0, max=6)
        v4 = v1 * v3
        v5 = v4 / 6
        v6 = self.bn(v5)
        v7 = self.globalpool(v6)
        v8 = v7.squeeze()
        v9 = self.fc(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
