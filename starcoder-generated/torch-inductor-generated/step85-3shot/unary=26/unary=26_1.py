
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(23, 255, 7, padding=0, bias=False)
    def forward(self, x):
        x3 = self.conv_t(x)
        x4 = x3 > 0
        x5 = x3 * 0.92
        x6 = torch.where(x4, x3, x5)
        return x6
# Inputs to the model
x = torch.randn(1, 23, 31, 51, 31)
