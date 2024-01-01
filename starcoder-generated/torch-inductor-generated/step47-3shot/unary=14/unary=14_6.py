
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(12, 273, 3, stride=(1, 2), padding=1, output_padding=(0, 0), groups=(4, 6), dilation=(4, 4))
    def forward(self, x1):
        v1 = self.conv_transpose_4(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(12, 4, 1, 1, 12)
