
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(97, 29, 3, stride=2, dilation=2, groups=1)
        self.conv_t = torch.nn.ConvTranspose2d(12, 294, 3, stride=2, dilation=2, groups=11, padding=0, bias=False)
    def forward(self, x):
        x1 = self.conv_t(self.conv(self.conv(self.conv(x))))
        x2 = self.conv(x1)
        x3 = x2 > 0
        x4 = x2 * -0.08
        x5 = torch.where(x3, x2, x4)
        return torch.nn.functional.relu(x5)
# Inputs to the model
x = torch.randn(4, 12, 14, 9)
