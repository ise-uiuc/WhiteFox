
class Model(torch.nn.Module):
    def __init__(self, kernel_size, padding):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(768, 624, kernel_size, stride=1, dilation=3, padding=padding)
        self.conv_t2 = torch.nn.ConvTranspose2d(624, 768, 1, stride=1)
    def forward(self, x1):
        x2 = self.conv_t1(x1)
        x3 = self.conv_t2(x2)
        x4 = x3 < 0
        x5 = x2 < 0
        x6 = x5 | x4
        x7 = x2.view(2304)
        return x6, x7 + x2 / 3.
kernel_size = 3
padding = 1
# Inputs to the model
x1 = torch.randn(16, 768, 108, 108)
