
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(256, 64, 3, padding=1, output_padding=0, groups=1, bias=False)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * 0.15853
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x = torch.randn(28, 256, 7, 1)
