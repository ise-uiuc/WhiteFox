
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(71, 8, (7,7,7), 1, 3, 1, bias=True)
    def forward(self, x14):
        y = self.conv_t(x14)
        y = y > 0
        y = y * -0.38
        w = torch.where(y, y, y)
        return w
# Inputs to the model
x14 = torch.randn(9, 71, 175, 1, 44)
