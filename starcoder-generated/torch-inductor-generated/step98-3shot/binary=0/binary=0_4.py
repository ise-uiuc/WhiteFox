
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 5, 1, stride=1, padding=1)
    def forward(self, x1, other, y, z=2, x2=3, p1=4, p2=5):
        v1 = self.conv(x1)
        if other == None:
            other = torch.nn.BatchNorm2d(v1.shape[1], affine=True)
        v2 = other(v1)
        v2 = v2 + y
        v2 = v2 + z
        v2 = v2 + x2
        v2 = v2 + p1
        v2 = v2 + p2
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
y1 = torch.randn(1, 5, 32, 32)
