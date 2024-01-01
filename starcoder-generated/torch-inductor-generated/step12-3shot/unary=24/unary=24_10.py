
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 2, 1, stride=4, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 3, 1, stride=4, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        negative_slope = 1
        v2 = self.conv2(v1)
        v3 = v2 > 0
        v4 = v2 * negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 56, 56)
