
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pointwise_conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=6, padding=4, ceil_mode=True, count_include_pad=False)
    def forward(self, x1):
        v1 = self.pointwise_conv(x1)
        v2 = self.avgpool(v1)
        v3 = torch.relu(v2)
        v4 = v3.permute(0, 2, 3, 1)
        v5 = v4.reshape(-1, 8)
        v6 = torch.sigmoid(v5)
        v7 = v6.reshape(v2.size(0), v3.size(1), v3.size(2), 1)
        v8 = v2 * v7
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
