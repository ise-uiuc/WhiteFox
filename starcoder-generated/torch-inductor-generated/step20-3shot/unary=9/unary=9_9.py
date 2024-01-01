
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        x2 = torch.randn(1, 8, 64, 64)
        v1 = self.conv(x1)
        v2 = (v1 + x2) * 3
        v3 = v2.clamp(min=0, max=6)
        v4 = torch.true_divide(v3, 6)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
