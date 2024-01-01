
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        v2 = self.conv(v1)
        v3 = self.conv(v2)
        v4 = v2 + x1
        v5 = torch.relu(v4)
        v6 = v3 + v5
        v7 = torch.relu(v6)
        v8 = v7 + x2
        v9 = torch.relu(v8)
        v10 = v9 + x3
        v11 = torch.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
