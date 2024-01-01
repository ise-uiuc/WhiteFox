
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_11 = torch.nn.ConvTranspose2d(1, 1, 7, stride=5, padding=0, output_padding=0, groups=1, dilation=2, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose_11(x1)
        v2 = torch.exp(v1)
        v3 = v1 / v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
