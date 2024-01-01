
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = v2 + x2 # this line is duplicated from v1
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 + v1 # this line is duplicated from v3
        v6 = torch.relu(v5)
        v7 = self.conv3(v5)
        v8 = v8 + x3
        v9 = v9 + v2
        v10 = v10 + v4
        v11 = torch.relu(v10)
        v12 = self.conv3(v9)
        v13 = v13 + v11
        v14 = v14 + x4
        v15 = v15 + v6
        v16 = torch.relu(v14)
        return v15
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
