
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(480, 7, 2, stride=2)
        self.conv = torch.nn.Conv2d(7, 7, 2, stride=2)
    def forward(self, x1):
        x2 = self.conv_t(x1)
        x3 = x2 > 0
        x4 = x2 * 0.5
        x5 = torch.where(x3, x2, x4)
        return x5 + torch.nn.functional.adaptive_avg_pool2d(x5, (1, 1)) + self.conv(x5)
# Inputs to the model
x1 = torch.randn(16, 480, 16, 16)
