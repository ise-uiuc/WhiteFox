
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=1, dilation=1)
    def forward(self, x3):
        x4 = self.conv_transpose(x3)
        x5 = x4 > 0
        x6 = x4 * 0.1
        x7 = torch.where(x5, x4, x6)
        return x7
# Inputs to the model
x3 = torch.randn(1, 1, 16, 16)
