
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        a1 = self.conv(x1)
        a2 = a1.add(1).clamp_min(0).clamp_max(1)
        a3 = (a2 - 0.5).div(0.5).add(0.5).mul(6)
        return a3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
