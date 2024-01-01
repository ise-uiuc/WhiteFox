
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1, dilation=1, groups=3)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.leaky_relu(v1)
        return v2
negative_slope = 1

# Input to the model
x1 = torch.randn(1, 3, 224, 224)
