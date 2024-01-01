
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 7, stride=2, padding=3)
        self.conv2 = torch.nn.Conv2d(1, 16, 5, stride=2, padding=2)
    def forward(self, x1):
        x2 = self.conv1(x1)
        v1 = self.conv2(x2)
        x4 = self.conv1(x2)
        v2 = self.conv2(x4)
        x6 = self.conv1(x2)
        v3 = self.conv2(x6)
        x7 = self.conv1(x2)
        v4 = self.conv1(x7)
        v5 = torch.relu(v1 + v2 + v3 + v4)
        v6 = self.conv2(v1)
        v7 = self.conv2(x4)
        v8 = self.conv2(x6)
        v9 = self.conv2(x7)
        v10 = torch.relu(v6 + v7 + v8 + v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
