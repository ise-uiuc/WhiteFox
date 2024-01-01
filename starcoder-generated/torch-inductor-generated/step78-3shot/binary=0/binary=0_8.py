
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
    def forward(self, x0, x1, other=4):
        v0 = self.conv(x0)
        v1 = self.conv(x1)
        v2 = v0 + v1
        v3 = v2 + other
        return v3
# Inputs to the model
x0 = torch.randn(1, 1, 64, 64)
x1 = torch.randn(1, 1, 32, 32)
