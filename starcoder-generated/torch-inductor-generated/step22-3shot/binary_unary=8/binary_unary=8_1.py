
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = self.conv(x1)
        v4 = v2 * v3
        v5 = v4
        v6 = v2 + v5
        v7 = torch.relu(v6)
        v8 = v1
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
