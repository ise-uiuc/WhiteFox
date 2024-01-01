
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(9, 11, 1)
    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        v2 = v1 + x2
        v3 = v2 + x3
        return v3
# Inputs to the model
x1 = torch.randn(1, 9, 64, 64)
x2 = torch.randn(4, 11, 64, 64)
x3 = torch.randn(4, 11, 64, 64)
