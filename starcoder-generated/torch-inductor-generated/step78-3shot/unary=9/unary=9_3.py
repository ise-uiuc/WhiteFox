
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 13, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3.0
        v3 = v2.clamp(min=0, max=6.0)
        v4 = v3 / (6+1)
        return v4
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
