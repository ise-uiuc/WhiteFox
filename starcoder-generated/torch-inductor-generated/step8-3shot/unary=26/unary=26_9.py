
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpo17n = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
    def forward(self, x16):
        x18 = self.conv_transpo17n(x16)
        x19 = x18 > 0
        x20 = x18 * 0.1
        x21 = torch.where(x19, x18, x20)
        return x21
# Inputs to the model
x16 = torch.randn(1, 1, 8, 8)
