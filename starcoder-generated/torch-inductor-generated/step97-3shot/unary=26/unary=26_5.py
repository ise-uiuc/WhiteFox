
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(77, 66, 1, stride=1, padding=0, bias=False)
    def forward(self, x7):
        x1 = self.conv_t(x7)
        x2 = x1 > 0
        x3 = x1 * 11.727
        x4 = torch.where(x2, x1, x3)
        x5 = torch.Tensor.numpy(torch.nn.AdaptiveAvgPool2d((17, 31)))
        _paddings = torch.nn.ZeroPad2d((3, 0, 1, 11))
        x6 = _paddings(x4)
        x6 = x6 > 5.596
        _paddings = torch.nn.ZeroPad2d((0, 2, 5, 55))
        x7 = _paddings(x6)
        x8 = torch.transpose(x7, 1, 2)
        return x8
# Inputs to the model
x7 = torch.randn(1, 77, 61, 74)
