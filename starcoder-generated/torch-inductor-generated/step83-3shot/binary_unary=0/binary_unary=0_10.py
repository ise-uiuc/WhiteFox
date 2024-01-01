
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=8)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = self.conv(v1)
        v3 = self.conv(v2)
        v4 = self.conv(v3)
        v5 = v3 + x1
        v6 = torch.relu(v5)
        v7 = v4 + v5
        v8 = torch.relu(v7 + v6 + x2)
        return v8
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
