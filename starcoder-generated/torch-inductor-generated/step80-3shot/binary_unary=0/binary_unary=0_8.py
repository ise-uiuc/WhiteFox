
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1 + x1
        v4 = torch.relu(v3)
        v5 = v2 + v4
        v6 = torch.relu(v5)
        v7 = v6 + x1
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 32, 64, 64)
