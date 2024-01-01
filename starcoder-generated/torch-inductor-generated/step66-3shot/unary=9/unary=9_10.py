
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1)
    def forward(self, x1):
        v2 = self.conv(x1) + 1
        v3 = v2.clamp(0, 6)
        v4 = v3 / 6
        return v4
# Inputs to the model
x1 = torch.randn(3, 3, 128, 128)
