
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 20, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1.transpose(-1, -2)
        v3 = v2 - 10
        v4 = F.relu(v3)
        v5 = v4.transpose(-1, -2)
        v6 = v5 + 1
        v7 = self.conv2(v6)
        v8 = v7 - 11
        v9 = F.relu(v8)
        v10 = self.conv2(v9)
        v11 = v10 + 1
        v12 = F.relu(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
