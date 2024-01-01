
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_3 = torch.nn.ConvTranspose3d(16, 16, 4, stride=2, padding=1, output_padding=0, groups=1, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose_3(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 10, 31, 33)
