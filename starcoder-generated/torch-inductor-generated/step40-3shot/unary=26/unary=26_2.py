
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t2 = torch.nn.ConvTranspose2d(35, 48, 5, stride=1, padding=0, bias=True)
        self.conv_t3 = torch.nn.ConvTranspose2d(35, 48, 6, stride=3, padding=0, bias=True)

    def forward(self, x10):
        f1 = self.conv_t2(x10)
        f2 = f1 > 0
        f3 = f1 * -0.132
        f4 = torch.where(f2, f1, f3)
        f5 = f4 + self.conv_t3(x10)
        return torch.nn.functional.adaptive_avg_pool2d(f5, (1, 1))
# Inputs to the model
x10 = torch.randn(5, 35, 26, 92)
