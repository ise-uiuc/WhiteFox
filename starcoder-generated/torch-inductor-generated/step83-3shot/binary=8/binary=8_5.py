
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu1 = torch.nn.ReLU()
    def forward(self, x1, x2):
        v1 = self.conv1(torch.add(x1, x2))
        v2 = self.conv1(torch.add(x1, x2))
        v3 = v1 + v2
        v4 = v3.add(v2)
        v5 = v3 + v4
        v6 = self.bn1(v3)
        v7 = v3 + v6
        v8 = self.relu1(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
