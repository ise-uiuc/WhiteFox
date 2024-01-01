
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        x2 = v1 + 3
        x3 = x2.clamp(0, 6)
        x4 = torch.div(x3, 6)
        return x4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
