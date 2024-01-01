
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv = torch.nn.modules.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv = torch.nn.Conv2d(32, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.Conv(x1)
        v2 = torch.add(v1, 3)
        v3 = torch.clamp(v2, 0, 6)
        v4 = torch.div(v3, 6)
        v5 = self.conv(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
