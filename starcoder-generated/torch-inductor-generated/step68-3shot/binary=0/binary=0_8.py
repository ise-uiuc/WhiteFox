
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 2, 1, stride=1, padding=1, groups=4)
    def forward(self, x1, x2, x3, other=None):
        v1 = self.conv(x1)
        v2 = x3 + 1
        v3 = v1 + v2
        v4 = v3 + x2
        final = v4 + x3
        return final
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
x2 = 1
x3 = 1
