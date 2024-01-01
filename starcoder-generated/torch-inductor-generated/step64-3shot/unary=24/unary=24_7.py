
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(25, 5, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(5, 5, 1, stride=1, padding=0)
    def forward(self, x):
        negative_slope = 0.325249
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = v2 > 0
        v4 = v2 * negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 25, 73, 48)
