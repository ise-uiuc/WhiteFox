
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = self.conv(x1)
        v4 = self.conv(x1)
        v5 = v1 + v2 + v3 + v4
        v6 = torch.relu(v5)
        v7 = self.conv(x1)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
