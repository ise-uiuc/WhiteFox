
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 1, stride=1, padding=7)
    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        v2 = v1 + x2
        v3 = x3 + v2
        v4 = torch.relu(v3)
        v5 = self.conv(v4)
        v6 = v5 + v2
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
