
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        v2 = self.conv(v1)
        v3 = v2 + x2
        v4 = torch.relu(v3)
        v5 = self.conv(v2)
        v6 = v5 + v4
        v7 = torch.relu(v6)
        v8 = v7 + x3
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(2, 16, 64, 64)
x2 = torch.randn(2, 16, 64, 64)
x3 = torch.randn(2, 16, 64, 64)
