
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(32)
        self.globalpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(32, 10)
        self.dropout = torch.nn.Dropout(0.2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.bn(v6)
        v8 = self.globalpool(v7)
        v9 = v8.squeeze()
        v10 = self.fc(v9)
        v11 = self.dropout(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
