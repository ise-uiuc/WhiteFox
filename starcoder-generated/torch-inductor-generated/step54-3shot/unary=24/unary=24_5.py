
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(14, 14, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(14, 19, 1, stride=1, padding=0)
    def forward(self, x):
        negative_slope = -6.520915
        v1 = self.conv1(x).permute(0, 1, 3, 2)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = v4.permute(0, 1, 3, 2)
        v6 = self.conv2(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 14, 63, 118)
