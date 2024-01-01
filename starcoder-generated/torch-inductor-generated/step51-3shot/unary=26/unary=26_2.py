
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(10, 10, 7, stride=1, padding=0)
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=[3, 3], stride=[1, 1], padding=0, dilation=1, ceil_mode=False)
        self.negative_slope = negative_slope
    def forward(self, x2):
        v1 = self.conv_t(x2)
        v4 = self.max_pool2d(v1)
        v6 = v4 > 0
        v7 = v4 * self.negative_slope
        v8 = torch.where(v6, v4, v7)
        return v8
negative_slope = -2.8388
# Inputs to the model
x2 = torch.randn(22, 10, 6, 6)
