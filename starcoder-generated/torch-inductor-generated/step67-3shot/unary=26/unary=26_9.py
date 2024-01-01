
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(163, 184, 2, stride=1, padding=0, dilation=2, output_padding=4, groups=3, bias=True)
    def forward(self, x28):
        h1 = self.conv_t(x28)
        return h1
# Inputs to the model
x28 = torch.randn(26, 163, 81, 32)
