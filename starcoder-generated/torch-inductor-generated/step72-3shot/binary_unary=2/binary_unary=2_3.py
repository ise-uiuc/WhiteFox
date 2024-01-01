
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(6, 10, 3, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(10, 16, 3, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(16, 3, 3, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 1
        v3 = F.relu(v2)
        v4 = v3 + 0.5
        v5 = self.conv2(v4)
        v6 = v5 - 2
        v7 = F.relu(v6)
        v8 = v7 + 1
        v9 = self.conv3(v8)
        v10 = v9 - 0.5
        v11 = F.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 6, 128, 128)
