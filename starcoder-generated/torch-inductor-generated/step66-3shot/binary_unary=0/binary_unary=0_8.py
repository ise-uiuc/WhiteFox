
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        v4 = v1 - self.conv3(v3)
        v5 = v1 + self.conv1(x1)
        v6 = v5 - v4
        v7 = v4 - v5
        v8 = v7 + v6
        v9 = v8 + v4
        v10 = v9 - x1
        v11 = self.conv1(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
