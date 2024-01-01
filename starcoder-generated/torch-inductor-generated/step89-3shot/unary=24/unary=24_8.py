
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(27, 63, 3, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(63, 256, 1, stride=1, padding=0)
    def forward(self, x):
        negative_slope = -1.6737899
        v1 = self.conv1(x)
        v2 = torch.transpose(v1, 3, 2)
        v3 = torch.transpose(v2, 2, 1)
        v4 = self.conv2(v3)
        v5 = v4 > 0
        v6 = v4 * negative_slope
        v7 = torch.where(v5, v4, v6)
        v8 = torch.transpose(v7, 1, 2)
        v9 = torch.transpose(v8, 2, 3)
        return v9
# Inputs to the model
x1 = torch.randn(1, 27, 13, 11)
