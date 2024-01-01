
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(9, 11, 1)
    def forward(self, x1, x2, other=None):
        v1 = self.conv(x1)
        v2 = v1 + x2
        if other == None:
            other = torch.randn(v1.shape)
        v3 = v2 + other
        return v3
# Inputs to the model
x1 = torch.randn(1, 9, 64, 64)
x2 = torch.randn(4, 11, 64, 64)
