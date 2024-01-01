
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(128, 256, 4, stride=2, bias=False)
    def forward(self, x3):
        x1 = self.conv_t(x3)
        x2 = x1 > 0
        x3 = x1 * -0.068
        x4 = torch.where(x2, x1, x3)
        x5 = torch.nn.functional.max_pool2d(torch.nn.functional.adaptive_avg_pool2d(x4, (1, 2)), kernel_size=3, stride=2, padding=0)
        x6 = x5 / 2.744
        return x5
# Inputs to the model
x3 = torch.randn(1, 128, 55, 100)
