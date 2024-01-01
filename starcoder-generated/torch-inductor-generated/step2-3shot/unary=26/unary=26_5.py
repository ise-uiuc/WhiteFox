
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        conv = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1, groups=1, dilation=1, output_padding=0, bias=True)
        v1 = conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
