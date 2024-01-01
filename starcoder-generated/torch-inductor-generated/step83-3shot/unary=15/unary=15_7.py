
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 64, 2, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 32, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.relu(v2)
        v4 = self.conv1(v3)
        v5 = self.conv2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv1(v6)
        v8 = self.conv2(v7)
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(64, 16, 35, 35)
