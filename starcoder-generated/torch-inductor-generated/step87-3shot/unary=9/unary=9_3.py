
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise = torch.nn.Conv2d(3, 4, 1, groups=4, stride=1, padding=0)
        self.Pointwise = torch.nn.Conv2d(8, 6, 1, groups=1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.depthwise(x1)
        v2 = self.Pointwise(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, 0, 6)
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
