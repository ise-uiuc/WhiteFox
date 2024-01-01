
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, (3, 3), stride=1, padding=0, dilation=1, groups=1)
        self.conv2 = torch.nn.Conv2d(8, 1, (1, 1), stride=1, padding=0, dilation=1, groups=1)
    def forward(self, x):
        negative_slope = 1.465364
        v1 = self.conv1(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v6 = self.conv2(v4)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 180, 106)
