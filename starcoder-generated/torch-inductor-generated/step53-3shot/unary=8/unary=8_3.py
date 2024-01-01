
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose1d(21, 31, 3, stride=4, padding=5, dilation=1, groups=1, output_padding=-5)
        self.conv_transpose2 = torch.nn.ConvTranspose1d(31, 50, 4, stride=5, padding=6, dilation=6, groups=13, output_padding=4)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v2 * v5
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 21, 17)
