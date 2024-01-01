
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v3 = self.conv(x1)
        v4 = v3 + 3
        v5 = v4.clamp_min(0)
        v6 = v5.clamp_max(6)
        v7 = v6.div(6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
