
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(28, 51, 3, stride=1, padding=1, bias=False)
    def forward(self, x16):
        _paddings = torch.nn.ConstantPad2d((0, 0, 0, 0), 0.0)
        x1 = _paddings(x16)
        x2 = self.conv_t(x1)
        x3 = x2 > 0
        x4 = x2 * -14483
        x5 = torch.where(x3, x2, x4)
        x6 = torch.nn.functional.adaptive_avg_pool2d(torch.nn.LogSigmoid()(x5), (1, 1))
        return x6
# Inputs to the model
x16 = torch.randn(1, 28, 3, 18)
