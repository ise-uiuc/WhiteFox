
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1, groups=3)
    def forward(self, x1, x2, other=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = torch.conv2d(x2, other, [1, 1], [1, 1], 0, [1, 1], 1)
        v3 = v2 + v1
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
