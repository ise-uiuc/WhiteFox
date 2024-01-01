
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv(x1)
        v2 = v1 - x2
        v3 = F.relu(v2)
        v4 = self.conv(v3)
        v5 = v4 - x3
        v6 = F.relu(v5)
        v7 = self.conv(v6)
        v8 = v7 - x4
        v9 = F.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 56, 56)
x2 = torch.randn(1, 3, 56, 56)
x3 = torch.randn(1, 3, 56, 56)
x4 = torch.randn(1, 3, 56, 56)
x5 = torch.randn(1, 3, 56, 56)
x6 = torch.randn(1, 3, 56, 56)
x7 = torch.randn(1, 3, 56, 56)
x8 = torch.randn(1, 3, 56, 56)
x9 = torch.randn(1, 3, 56, 56)
