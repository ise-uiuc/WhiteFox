
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        a1 = v1 + x1
        v5 = a1 + x2
        v6 = self.conv3(v5)
        v7 = torch.relu(v6)
        v8 = v4 + v1
        v9 = v6 + x2
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
