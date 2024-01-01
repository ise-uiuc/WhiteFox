
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 + x1
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 + v3
        v6 = torch.relu(v5)
        v7 = self.conv1(v6)
        v8 = v7 + v6
        v9 = torch.relu(v8)
        v10 = self.conv2(v9)
        v11 = v10 + v9
        v12 = torch.relu(v11)
        return v12
# Inputs to the model
x1 = torch.randn(2, 16, 64, 64)
