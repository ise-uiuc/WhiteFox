
class Model(torch.nn.Module):
    def __init__(self, min):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 2, stride=2, padding=0)
        self.min = min
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.nn.functional.softplus(v1, beta=-2.0)
        v3 = torch.clamp_min(v2, self.min)
        return v3
min = -0.2
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
