
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(256, 4, 2, dilation=2, stride=1, padding=1)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * -4.94
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x = torch.randn(1, 256, 1, 1)
