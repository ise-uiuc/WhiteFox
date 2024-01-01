
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, groups=1)
        self.scale = torch.nn.Parameter(torch.tensor([1.0]))
    def forward(self, x1, other=1):
        v1 = self.conv(x1)
        v2 = v1.mul(self.scale) + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
