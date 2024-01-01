
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(29, 70, 25, stride=1, padding=10, output_padding=3, groups=14)
    def forward(self, x1):
        x1 = self.conv_t(x1)
        x2 = x1 > 0
        x3 = x1 * 0.62
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.adaptive_avg_pool2d(x4, (1, 1))
# Inputs to the model
x1 = torch.randn(9, 29, 14, 1)
