
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 6, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + x2
        v4 = torch.relu(v3)
        a1 = self.conv1(x3)
        a2 = self.conv1(x3)
        v5 = torch.relu(v4)
        a3 = torch.relu(a1 + a2)
        v6 = torch.relu(v5 + a3)
        v7 = v6 + x3
        v8 = self.conv2(v7)
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
