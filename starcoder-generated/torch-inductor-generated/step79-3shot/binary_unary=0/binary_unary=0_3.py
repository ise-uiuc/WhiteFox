
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.cat([v1, v1, v1], axis=0)
        v3 = torch.cat([v2, v2, v2], axis=0)
        v4 = torch.add(v3, v3)
        v5 = torch.relu(v4)
        v6 = self.conv2(v5)
        v7 = torch.add(v6, v6)
        v8 = torch.relu(v7)
        v9 = self.conv3(v8)
        v10 = torch.add(v9, v9)
        return torch.relu(v10)
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
