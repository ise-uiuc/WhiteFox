
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bmm1 = torch.nn.Bilinear(8, 8, 8)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = v2.softmax(dim=1)
        v4 = x2.bmm(v3).relu()
        v5 = self.bmm1(v1, v4)
        return v5
# Inputs to the model
x1 = torch.randn(16, 3, 32, 32)
x2 = torch.randn(8, 3, 16, 16)
