
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=2, dilation=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = self.sigmoid(v2)
        v4 = v1 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
