
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(27, 69, 1, stride=1, padding=1, bias=False)
    def forward(self, x3):
        _paddings = torch.nn.ZeroPad2d((0, 0, 1, 1))
        x1 = _paddings(x3)
        x2 = self.conv_t(x1)
        x3 = x2 > 0
        x4 = x2 * -12.022
        x5 = torch.where(x3, x2, x4)
        x6 = torch.nn.functional.adaptive_avg_pool2d(torch.nn.Softplus()(x5), (1, 3))
        x7 = torch.transpose(x6, 2, 3)
        return x7
# Inputs to the model
x3 = torch.randn(1, 27, 81, 61)
