
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(195, 1, 3, stride=1, padding=1, bias=False)
        self.negative_slope = negative_slope
    def forward(self, x4):
        f1 = self.conv_t(x4)
        f2 = f1 > 0
        f3 = f1 * self.negative_slope
        f4 = torch.where(f2, f1, f3)
        return f4
# Inputs to the model
x4 = torch.randn(41, 195, 17, 20)
negative_slope = torch.randn(())
