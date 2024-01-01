
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, add=False):
        v1 = self.conv1(x2)
        v2 = v1 + x1
        v4 = v1 + x2 + 3
        v5 = v1 + x1 + 5
        v6 = v1 - x1 - 3
        v7 = v1 - x2 - 3
        v3 = torch.relu(v2)
        if add:
            v8 = v3 + x1
        else:
            v8 = v3 + v4
        v9 = torch.relu(v8)
        v10 = self.conv3(v9)
        if add:
            v11 = v10 + v5
        else:
            v11 = v10 + v6
        v12 = self.conv2(v7)
        v13 = v12 + v10
        v14 = torch.relu(v13)
        return v14
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
