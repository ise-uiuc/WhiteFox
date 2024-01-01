
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 18, [1, 2, 1], stride=[2, 1, 1], padding=[1, 1, 0])
    def forward(self, x1, other):
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
other = torch.randn(1, 2, 32, 64)
