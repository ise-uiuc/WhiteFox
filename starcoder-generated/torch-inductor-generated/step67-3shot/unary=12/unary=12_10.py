
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.detach()
        v3 = v2.sigmoid()
        v4 = v1 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 7, 7)
x3 = torch.randn(1, 3, 64, 64)
x4 = torch.randn(1, 3, 28, 28)
x5 = torch.randn(1, 3, 64, 64)
x6 = torch.randn(1, 3, 56, 56)
