
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 3, stride=2, padding=2, groups=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 2.0
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
