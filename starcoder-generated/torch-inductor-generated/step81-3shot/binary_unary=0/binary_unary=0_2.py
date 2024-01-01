
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        a1 = self.conv(x3)
        v2 = v1 + x3
        v3 = torch.hardtanh(v2)
        v4 = a1 + x2
        v5 = torch.relu(v4)
        v6 = v3 + v5
        v7 = torch.relu(v6)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
x2 = torch.randn(1, 1, 64, 64)
x3 = torch.randn(1, 1, 64, 64)
