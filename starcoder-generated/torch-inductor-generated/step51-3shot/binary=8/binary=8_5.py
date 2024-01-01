
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = self.bn1(v2)
        v4 = self.bn2(v2)
        v5 = v3.sub(v4)
        v6 = self.bn1(v5)
        v7 = self.bn2(v5)
        v8 = v6.add(v7)
        v9 = self.bn1(v8)
        v10 = self.bn2(v8)
        v11 = v9.div(v10)
        v12 = self.bn1(v11)
        v13 = self.bn2(v11)
        v14 = v12.mul(v13)
        return v14
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
