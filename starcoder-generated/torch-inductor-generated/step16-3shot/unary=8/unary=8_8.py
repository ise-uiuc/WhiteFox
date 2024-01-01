
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv4 = torch.nn.Conv2d(32, 24, 3, stride=2)
        self.conv3 = torch.nn.Conv2d(24, 16, 3, stride=2)
    def forward(self, x1):
        v1 = self.conv4(x1)
        v2 = self.conv3(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v2 * v5
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
