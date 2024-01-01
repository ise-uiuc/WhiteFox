
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.nn.functional.interpolate(self.conv(x1), scale_factor = 1.0)
        v2 = torch.clamp(v1 + 3, min = 0)
        v3 = self.conv(x1)
        return v2 + 1.0 + v3
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
