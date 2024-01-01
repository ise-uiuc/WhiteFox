
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d((41), 1, 1, stride=1, padding=0, bias=False)
    def forward(self, x3):
        x4 = self.conv_t(x3)
        x5 = x4 > 0
        x6 = x4 * -0.4
        x7 = torch.where(x5, x4, x6)
        x8 = x7 * -0.8
        x9 = x8 + 0.8
        x10 = torch.round(x9)
        return x10
# Inputs to the model
x3 = torch.randn(1, 41, 14, 14)
