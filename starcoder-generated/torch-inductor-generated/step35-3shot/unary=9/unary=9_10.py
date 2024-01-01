
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = ((3 + v1.sum()) - v1.mean())
        v3 = v2 * v1.var()
        v4 = v3.relu_().rsqrt_()
        v5 = self.conv(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
