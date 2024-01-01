
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.relu(x)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = v5 + v2
        v7 = torch.relu(v6)
        v8 = torch.relu(v4)
        v9 = v8 + v3
        v10 = torch.relu(v9)
        v11 = self.conv3(v10)
        v12 = v11 + v3
        v13 = torch.relu(v12)
        return v13
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
