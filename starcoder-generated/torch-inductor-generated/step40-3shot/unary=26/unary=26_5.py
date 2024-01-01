
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(74, 218, 4, stride=1, padding=1, bias=True)
    def forward(self, x23):
        f1 = self.conv_t(x23)
        f2 = f1 > 0
        f3 = f1 * -0.140
        f4 = torch.where(f2, f1, f3)
        return torch.nn.functional.adaptive_avg_pool2d(f4, (1, 1))
# Inputs to the model
x23 = torch.randn(48, 74, 14, 36)
