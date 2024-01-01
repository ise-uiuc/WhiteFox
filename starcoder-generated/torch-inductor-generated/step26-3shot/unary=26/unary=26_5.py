
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(8, 21, 3, stride=2, padding=1, dilation=3, output_padding=1)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = v1 > 0
        v3 = v1 * 0.012
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(18, 8, 8, 8)
