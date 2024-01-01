
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = v5 + self.conv2(x1)
        v7 = self.conv2(v4)
        v8 = self.conv2(x1)
        v9 = torch.relu(v6 + v7)
        v10 = self.conv2(x1)
        v11 = torch.relu(v10)
        v12 = v9 + v11
        return v12
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
