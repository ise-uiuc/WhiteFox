
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(29, 6, 4, stride=4, padding=2, output_padding=1, bias=False)
    def forward(self, x):
        x10 = self.conv_t(x)
        x11 = x10 > 0
        x12 = x10 * 0.698
        x13 = torch.where(x11, x10, x12)
        return x13
# Inputs to the model
x = torch.randn(1, 29, 10, 12)
