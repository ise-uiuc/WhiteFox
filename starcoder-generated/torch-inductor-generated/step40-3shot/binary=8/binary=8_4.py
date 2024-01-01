
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.bn1(v1)
        v4 = self.bn2(v2)
        v5 = v3 + v4
        v6 = self.conv3(v5)
        v7 = self.conv4(v5)
        v8 = v6 + v7
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
