
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(89, 1, 4, stride=4, padding=2, groups=39, bias=True)
    def forward(self, x11):
        g1 = self.conv_t(x11)
        g2 = g1 > 0
        g3 = g1 * -9.7
        g4 = torch.where(g2, g1, g3)
        return torch.nn.functional.adaptive_avg_pool2d(g4, (78, 132))
# Inputs to the model
x11 = torch.randn(209, 89, 7, 40)
