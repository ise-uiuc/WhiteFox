
class Model(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, in_size, out_size, bias=None, dilation=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(2, 2, 1, stride=2, padding=2)
    def forward(self, x1=None, other=None):
        v1 = self.conv(x1)
        y1 = self.conv2(other)
        v2 = v1 + y1
        return v2
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
