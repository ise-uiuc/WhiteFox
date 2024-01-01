
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.linear1 = torch.nn.Linear(16, 16)
        self.linear2 = torch.nn.Linear(16, 32)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 + x2
        v4 = self.conv3(v3)
        v5 = self.conv4(v4)
        v6 = v4 * 2
        v7 = v4 - v5
        v8 = v7 * v4
        v9 = v4 - v6
        v10 = v9 * v4
        v11 = self.linear1(v8)
        v12 = self.linear2(v10)
        v13 = self.linear2(v11)
        v14 = v13 + v12
        return v14
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
