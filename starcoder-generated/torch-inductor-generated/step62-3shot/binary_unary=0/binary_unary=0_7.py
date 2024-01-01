
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        v2 = v1 + x3
        v3 = torch.relu(v2)
        v4 = torch.neg(v3)
        v5 = v4 + v1
        v6 = torch.relu(v5)
        return torch.neg(v6)
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
