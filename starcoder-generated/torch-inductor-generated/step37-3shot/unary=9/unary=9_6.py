
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 9, 1)
    def forward(self, x1):
        v1 = torch.clamp_max(3 + self.conv(x1), 6)
        v2 = v1 / 6
        return v2
# Inputs to the model
x1 = torch.randn(3, 3, 64, 64)
