
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(180, 56, 4, stride=2, padding=0, bias=True)
    def forward(self, x2):
        f1 = self.conv_t(x2)
        f2 = f1 > 0
        f3 = f1 * -0.165
        f4 = torch.where(f2, f1, f3)
        return torch.nn.functional.adaptive_avg_pool2d(f4, (1, 1))
# Inputs to the model
x2 = torch.randn(41, 180, 13, 15)
