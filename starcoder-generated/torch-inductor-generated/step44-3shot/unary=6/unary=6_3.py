
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, 0, 6)
        v4 = v1 * v3
        v5 = v4 / 6
        return torch.mean(v5, dim=[2, 3])
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
