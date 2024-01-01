
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = v1 + v2
        v4 = torch.clamp(v3, min=0, max=6)
        v5 = torch.clamp(v4, max=6)
        v6 = torch.div(v5, 6)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
