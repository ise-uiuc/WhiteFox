
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        b = 5
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = x3 + v1
        v4 = torch.relu(v3)
        v5 = v2 + v4
        v6 = torch.nn.functional.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
