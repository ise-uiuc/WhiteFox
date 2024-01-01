
class Model(torch.nn.Module):
def __init__(self, negative_slope=0.1):
    super().__init__()
    self.conv = torch.nn.Conv2d(8, 8, 1, stride=2)
    self.negative_slope = negative_slope
def forward(self, x1):
    t1 = self.conv(x1)
    t2 = t1 > 0
    t3 = t1 * self.negative_slope
    t4 = torch.where(t2, t1, t3)
    return t4
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
