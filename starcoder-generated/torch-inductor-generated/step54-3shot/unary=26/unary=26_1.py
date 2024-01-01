
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(16, 8, 10, 5, 1, bias=False)
    def forward(self, x9):
        x1 = self.conv_t(x9)
        x2 = x1 > 0
        x3 = x1 * -0.24999999999999999
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x9 = torch.randn(3, 16, 5, 5)
