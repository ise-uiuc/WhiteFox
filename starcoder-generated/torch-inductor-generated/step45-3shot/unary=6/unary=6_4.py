
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3, stride=1, padding=0)
        self.flatten = torch.nn.Flatten()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = torch.clamp(v2, 0, 6)
        v4 = v1 * v3
        v5 = v4 / 6
        v6 = self.flatten(v4)
        return torch.sigmoid(v6)
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
