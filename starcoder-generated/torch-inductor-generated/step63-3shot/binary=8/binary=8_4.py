
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 5, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = v1 + v1
        v4 = v2.mul(v3)
        v5 = v4.div(v1.div(v4 + v3))
        v6 = v3.mul(v4)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
