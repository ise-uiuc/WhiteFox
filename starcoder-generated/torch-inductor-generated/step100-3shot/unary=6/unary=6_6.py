
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 20
        v3 = torch.clamp(v2, min=-10, max=5)
        v4 = torch.clamp(v1, min=-10, max=5)
        v5 = v4 * v3 / 50
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
