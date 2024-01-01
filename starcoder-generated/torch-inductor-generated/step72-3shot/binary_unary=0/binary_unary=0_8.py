
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, sub):
        v1 = self.conv1(x1)
        v2 = x2 - 100
        v3 = torch.relu(v2)
        if sub:
            v4 = v2 - v1
        else:
            v4 = 100 - v3
        v5 = v4 + v3
        v6 = v5 + v1
        v7 = torch.relu(v6)
        v8 = x2 * self.conv2(v7)
        v9 = v8 * v7
        v10 = torch.relu(v9)
        v11 = self.conv3(v10)
        v12 = v11 + v8
        v13 = torch.relu(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
