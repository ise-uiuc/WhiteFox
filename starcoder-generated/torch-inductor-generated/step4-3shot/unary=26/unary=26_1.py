
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(1, 1, 3, stride=2, padding=1, bias=False, dilation=1, groups=1, output_padding=0)
    def forward(self, x0):
        v5 = self.conv_transpose(x0)
        v6 = v5 > 3
        v7 = v5 * 6
        v8 = torch.where(v6, v5, v7)
        return v8
# Inputs to the model
x0 = torch.randn(1, 1, 3, 3, 3)
