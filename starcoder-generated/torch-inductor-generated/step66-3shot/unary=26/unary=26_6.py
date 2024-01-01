
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(960, 32, 3, stride=2, padding=1, bias=False)
        self.conv2d = torch.nn.Conv2d(32, 224, 3, stride=1, padding=0, groups=2, bias=False)
    def forward(self, x9):
        x1 = self.conv_t(x9)
        x2 = self.conv2d(x1)
        x3 = x2 > 0
        x4 = x2 * -4.94
        x5 = torch.where(x3, x2, x4)
        return x5
# Inputs to the model
x9 = torch.randn(1, 960, 42, 35)
