
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, dilation=1)
    def forward(self, x1, v1=None, d1=None, o1=None, o2=None):
        v2 = self.conv(x1)
        if v2.shape!= v1.shape:
            v1 = torch.randn(v1.shape)
        v3 = v1 + v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
v1 = torch.randn(1, 1000, 28, 28)
