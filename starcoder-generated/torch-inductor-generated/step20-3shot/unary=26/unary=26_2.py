
class Model(torch.nn.Module):
    def __init__(self, stride, padding, dilation):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(int(480 * 7 / 8 + 0.5), 7, 2, stride=stride, padding=padding, dilation=dilation)
    def forward(self, x1):
        x2 = self.conv_t(x1)
        x3 = x2
        x4 = x3 > 0
        x5 = x3 * 0.5
        x6 = torch.where(x4, x3, x5)
        return x6
stride = 2
padding = 1
dilation = 1
# Inputs to the model
x1 = torch.randn(32, 480, 16, 16)
