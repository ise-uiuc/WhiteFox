
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(28, 6, 1, stride=1, padding=1, bias=False)
    def forward(self, x24):
        f1 = self.conv_t(x24)
        f2 = f1 > 0
        f3 = f1 * -0.189
        f4 = torch.where(f2, f1, f3)
        return torch.nn.functional.adaptive_avg_pool2d(f4, (1, 4))
# Inputs to the model
x24 = torch.randn(4, 28, 17, 34)
