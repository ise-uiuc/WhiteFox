
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(15, 28, 8, stride=3, padding=3, dilation=2, groups=3)
    def forward(self, x3):
        x1 = self.conv_t(x3)
        x2 = x1 < 0.3
        x3 = x1 * 2
        x4 = torch.where(x2, x1, x3)
        return torch.tanh(x4)
# Inputs to the model
x3 = torch.randn(11, 15, 6, 5)
