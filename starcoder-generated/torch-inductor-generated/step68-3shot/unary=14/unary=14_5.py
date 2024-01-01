
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_13 = torch.nn.ConvTranspose2d(2, 2, 4, stride=2, padding=1, dilation=1, output_padding=0, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose_13(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 16, 16)
