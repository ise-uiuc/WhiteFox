
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x5, x6):
        v1 = self.conv1(x5)
        v2 = self.conv2(x5)
        v3 = v1 + x6
        v4 = v2 + x6
        v5 = torch.relu(v3)
        v6 = torch.relu(v4)
        v7 = v5
        v8 = v6 + x6
        v9 = torch.relu(v8)
        v10 = v7 - x6
        v11 = torch.relu(v10)
        return v9 + v11
# Inputs to the model
x5 = torch.randn(1, 16, 64, 64)
x6 = torch.randn(1, 16, 64, 64)
