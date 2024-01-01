
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.bn1(v1)
        v3 = v1 + v2
        return v1 + v2 + v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
