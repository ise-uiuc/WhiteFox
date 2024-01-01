
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu1 = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = self.relu1(v3)
        v5 = self.bn1(v3)
        v6 = self.conv1(x2)
        v7 = self.conv2(x1)
        v8 = v6 + v7
        v9 = v5.add(v8)
        v10 = self.sigmoid(v9)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 3, 224, 224)
