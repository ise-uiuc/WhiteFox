
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1) + 3
        v4 = torch.clamp(v1, max=6).div(6)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
