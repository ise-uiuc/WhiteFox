
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 2, stride=3, padding=4)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0, max=6)
        v4 = v1 * v3
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(2, 3, 113, 113)
