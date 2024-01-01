
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x):
        negative_slope = 0.2
        v1 = self.conv1(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.conv2(v4)
        return v5
v1 = (torch.randn((1, 3, 5, 5)), torch.randn((1, 3, 5, 5))),
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 5, 5)
x3 = torch.randn((1, 3, 5, 5)), torch.randn((1, 3, 5, 5))
