
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 76, 3, stride=2, padding=1, dilation=1, groups=76, output_padding=0)
    def forward(self, x1, x2, x3, x4, x5):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6, x2, x3, x4, x5
# Inputs to the model (Note: the size of inputs may differ)
x1 = torch.randn(1, 8, 36, 29)
x2 = torch.randn(1, 8, 25, 17)
x3 = torch.randn(1, 8, 48, 12)
x4 = torch.randn(1, 8, 24, 64)
x5 = torch.randn(1, 8, 9, 6)
