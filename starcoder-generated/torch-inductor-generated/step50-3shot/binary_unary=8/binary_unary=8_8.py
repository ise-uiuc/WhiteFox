
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 + self.conv2(x1)
        v3 = torch.relu(v2)
        v4 = v1 + self.conv2(x1)
        v5 = torch.relu(v4)
        v6 = self.conv2(x1)
        v7 = v3 + v6
        v8 = self.conv2(x1)
        v9 = v5 + v8
        v10 = torch.relu(v9)
        v11 = self.conv2(x1)
        v12 = v7 + v11
        v13 = torch.relu(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
