
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(12, 103, 1, stride=1, padding=0, bias=False)
    def forward(self, x1):
        h1 = self.conv_t(x1)
        h2 = h1 > 0
        h3 = h1 * 0.785
        h4 = torch.where(h2, h1, h3)
        return torch.nn.functional.adaptive_avg_pool2d(h4, (1, 1))
# Inputs to the model
x1 = torch.randn(1, 12, 40, 28)
