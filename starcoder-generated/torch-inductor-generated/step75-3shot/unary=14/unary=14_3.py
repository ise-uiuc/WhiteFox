
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_8 = torch.nn.ConvTranspose3d(22, 15, 3, stride=3, padding=0,
                                                     output_padding=(1, 0, 1))
    def forward(self, x1):
        v1 = self.conv_transpose_8(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 22, 8, 6, 7)
