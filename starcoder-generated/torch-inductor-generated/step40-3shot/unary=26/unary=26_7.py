
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(12, 3, 4, stride=2, padding=0, bias=True)
    def forward(self, x3):
        f1 = self.conv_t(x3)
        f2 = f1 > 0
        f3 = f1 * -0.042
        f4 = torch.where(f2, f1, f3)
        return torch.nn.functional.adaptive_avg_pool2d(f4, (1, 1))
# Inputs to the model
x3 = torch.randn(19, 12, 2, 63)
