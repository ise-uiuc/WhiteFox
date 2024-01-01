
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=1)
    def forward(self, x8):
        x9 = self.conv_transpose(x8)
        x10 = x9 > 0
        x11 = x9 * -2
        x12 = torch.where(x10, x9, x11)
        return x12
# Inputs to the model
x8 = torch.randn(1,1,32)
