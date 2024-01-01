
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(16, 6, 2, stride=1)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = self.conv_t(x)
        x3 = torch.mul(1.7862, x2)
        x4 = x3 < 0
        x5 = x3 * 1625928689.0
        x6 = torch.where(x4, x3, x5)
        return torch.mul(2397.630, x1) + torch.mul(159.033, x6)
# Inputs to the model
x = torch.randn(23, 16, 128, 75)
