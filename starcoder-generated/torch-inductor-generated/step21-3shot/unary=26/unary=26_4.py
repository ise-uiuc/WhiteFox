
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(20, 8, 1, stride=1, padding=1, bias=False)
    def forward(self, x3):
        x4 = self.conv_t(x3)
        x5 = x4 > 0
        x6 = x4 * 0.2019
        x7 = torch.where(x5, x4, x6)
        return x7
# Inputs to the model
x3 = torch.randn(1, 20, 9, 9)
