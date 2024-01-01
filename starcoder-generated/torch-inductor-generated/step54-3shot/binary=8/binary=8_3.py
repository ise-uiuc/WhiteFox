
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.bn1(v1)
        v4 = self.bn2(v2)
        v5 = v3 + v4
        v6 = v1.transpose()
        v7 = v6.add(v4)
        v8 = v3.bmm(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 2, 10, 10)
x2 = torch.randn(1, 2, 10, 10)
