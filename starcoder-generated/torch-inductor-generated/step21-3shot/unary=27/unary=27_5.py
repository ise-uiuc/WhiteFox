
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, 1, stride=2, padding=6)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, min=0.)
        v3 = torch.clamp_max(v2, max=1.)
        return v3
# Inputs to the model
x1 = torch.randn(1, 32, 100, 10, 100)
