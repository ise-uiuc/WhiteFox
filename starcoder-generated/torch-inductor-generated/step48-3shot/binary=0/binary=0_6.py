
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, stride=1)
    def forward(self, x1, p1=None, z=None):
        v1 = self.conv(x1)
        if p1 is None:
            p1 = torch.randn(v1.shape)
        v2 = v1 + p1
        if z is not None:
            v2 = v2 + z
        return p1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
