
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = v2.min(3.0).item()
        v4 = v2.max(4.0).item()
        v5 = v2.clamp(min=0, max=6).div(6.0)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
