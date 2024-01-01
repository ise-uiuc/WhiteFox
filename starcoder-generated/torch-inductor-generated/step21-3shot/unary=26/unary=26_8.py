
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(7, 12, 2, stride=1, bias=False)
        self.conv_t = torch.nn.ConvTranspose2d(12, 12, 2, stride=1)
    def forward(self, x7):
        x8 = self.conv2d(x7)
        x9 = self.conv_t(x8)
        x10 = x9 > 0
        x11 = x9 * 0.3
        x12 = torch.where(x10, x9, x11)
        return x12
# Inputs to the model
x7 = torch.randn(1, 7, 32, 32)
