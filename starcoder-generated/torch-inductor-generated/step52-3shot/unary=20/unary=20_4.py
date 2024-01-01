
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        stride = 2
        kernel_size = 3
        dilation = 1
        groups = 1
        padding = ((stride - 1) // 2, dilation * (kernel_size - 1) // 2)
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)
