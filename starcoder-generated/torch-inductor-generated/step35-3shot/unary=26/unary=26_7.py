
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(1, 32, 2, padding=1, dilation=2, output_padding=1)
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = v1 > 0
        v3 = v1 * 1.0
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x = torch.randn(1, 1, 6, 12, 14)
