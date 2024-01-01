
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(v1)
        v6 = torch.relu(v5)
        v7 = self.conv(v6)
        v8 = v7 - v1
        v9 = torch.relu(v8)
        v10 = torch.relu(v9)
        v11 = self.conv2(v10)
        return v11
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
# Model begins