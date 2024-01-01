
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(397, 28, 4, stride=3, padding=1, output_padding=1, bias=True)
    def forward(self, x18):
        v1 = self.conv_t(x18)
        v2 = torch.round(v1)
        return v2
# Inputs to the model
x18 = torch.randn(1, 397, 30, 26)
