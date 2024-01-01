
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(40, 7, 3, stride=4, padding=1, output_padding=1, groups=10)
    def forward(self, x3):
        g1 = self.conv_t(x3)
        g2 = g1 > 0
        g3 = g1 * -0.01
        g4 = torch.where(g2, g1, g3)
        return torch.nn.functional.adaptive_avg_pool2d(torch.nn.ReLU()(g4), (7, 21))
# Inputs to the model
x3 = torch.randn(43, 40, 3, 41)
