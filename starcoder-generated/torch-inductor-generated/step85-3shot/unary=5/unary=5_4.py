
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool2d = torch.nn.MaxPool2d(13, stride=9, padding=9)
        self.conv2d = torch.nn.Conv2d(13, 13, 6, stride=5, dilation=5, padding=15)
    def forward(self, x1):
        v1 = self.max_pool2d(x1)
        v2 = self.conv2d(v1)
        v3 = v2 * v2
        v4 = torch.rsqrt(v3)
        v5 = v1 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 13, 45, 45)
