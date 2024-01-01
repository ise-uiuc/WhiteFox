
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(73, 43, 4, stride=1, padding=1, output_padding=0, bias=False)
    def forward(self, x8):
        v1 = self.conv_t(x8)
        v2 = v1 > 0
        v3 = v1 * -0.219
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x8 = torch.randn(1, 73, 30, 38)
