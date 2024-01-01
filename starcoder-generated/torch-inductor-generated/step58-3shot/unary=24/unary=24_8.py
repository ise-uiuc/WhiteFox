
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(6, 14, 8, stride=2, padding=3)
        self.conv2 = torch.nn.Conv2d(14, 20, 9, stride=2, padding=-1)
    def forward(self, x):
        negative_slope = -0.93233576
        v1 = self.conv1(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        u1 = self.conv2(v4)
        w1 = u1 > 0
        t1 = u1 * negative_slope
        t2 = torch.where(w1, u1, t1)
        return t2
# Inputs to the model
x1 = torch.randn(1, 6, 362, 81)
