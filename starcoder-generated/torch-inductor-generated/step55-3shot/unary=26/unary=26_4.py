
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(3000, 1020, (2, 4, 6), stride=(1, 2, 3),
                                                padding=1, bias=False)
    def forward(self, x2):
        x1 = self.conv_t(x2)
        x2 = x1 > 0
        x3 = x1 * 0.9125
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x2 = torch.randn(23, 3000, 7, 8, 6)
