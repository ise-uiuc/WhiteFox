
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(17, 64, 8, stride=1, padding=0)
    def forward(self, x8):
        x9 = self.conv_t(x8)
        x10 = x9 - 0.5
        x11 = x10 * 1.45
        x12 = torch.round(x11)
        return x12
# Inputs to the model
x8 = torch.randn(1, 17, 4, 4)
