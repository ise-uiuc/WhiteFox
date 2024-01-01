
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(12, 8, 3, stride=2, padding=0, bias=False)
    def forward(self, x3):
        x4 = self.conv_t(x3)
        x5 = x4 > 0
        x6 = x4 * -1
        x7 = torch.where(x5, x4, x6)
        return x7
# Inputs to the model
x3 = torch.randn(1, 12, 16, 32, 48)
