
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 25, (1, 1), stride=(1, 1))
        self.conv2 = torch.nn.Conv2d(25, 42, (1, 1), stride=(1, 1), bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.nn.functional.leaky_relu(v1, negative_slope=0.010000000000000009)
        v3 = torch.nn.functional.max_pool2d(v2, [3, 3], stride=[2, 2], padding=1, dilation=1, ceil_mode=False)
        v4 = self.conv2(torch.nn.functional.interpolate(v3, [6, 6], mode='nearest'))
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 564, 10)
