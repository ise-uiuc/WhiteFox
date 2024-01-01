
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x2)
        v2 = self.conv2(x1)
        v3 = self.conv3(x2)
        v4 = v1 + x1
        v5 = self.conv1(v4)
        v6 = v5 + v1
        v7 = torch.relu(v6)
        v8 = v2 + x1
        v9 = self.conv2(v8)
        v10 = v9 + v2 + x3
        v11 = torch.relu(v10)
        v12 = v3 + x2
        v13 = self.conv3(v12)
        v14 = v11 + torch.relu(v13)
        return v14
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
