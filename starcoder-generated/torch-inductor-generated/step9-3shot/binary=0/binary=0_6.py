
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2, other=None):
        v1 = self.conv(x1)
        if other is not None:
            v2 = v1 + other
        else:
            v2 = torch.randn(v1.shape)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
