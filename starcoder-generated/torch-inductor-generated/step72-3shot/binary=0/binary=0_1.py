
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
    def forward(self, x, other=None):
        v1 = self.conv(x)
        if not other is None:
            v2 = v1 + other
        else:
            v2 = v1
        return v2
# Inputs to the model
x = torch.randn(1, 1, 24, 24)
