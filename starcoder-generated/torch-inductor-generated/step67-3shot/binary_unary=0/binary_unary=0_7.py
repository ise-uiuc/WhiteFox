
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.avgpool = torch.nn.AvgPool2d(5, stride=1)
    def forward(self, x1, x2, add=False):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = v2 + x1
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = v5 + v5
        v7 = torch.relu(v6)
        v8 = self.conv3(v7)
        v9 = self.conv2(self.conv1(x2)) + v8
        v10 = torch.relu(v9)
        v11 = self.avgpool(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
