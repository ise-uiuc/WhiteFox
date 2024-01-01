
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(455, 287, 8, stride=4, padding=0, bias=True)
    def forward(self, x11):
        h1 = self.conv_t(x11)
        h2 = h1 > 0
        h3 = h1 * -0.0896
        h4 = torch.where(h2, h1, h3)
        return torch.nn.functional.adaptive_avg_pool2d(h4, (1, 1))
# Inputs to the model
x11 = torch.randn(11, 455, 16, 12)
