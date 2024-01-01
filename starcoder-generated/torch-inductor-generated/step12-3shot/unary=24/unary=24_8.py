
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(8, 32, 4, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(32, 32, 2, stride=1, padding=1)
    def forward(self, x):
        negative_slope = 1
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = v2 > 0
        v4 = v2 * negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5
# Inputs to the model
x1 = torch.randn(4, 8, 56, 56, 56)
