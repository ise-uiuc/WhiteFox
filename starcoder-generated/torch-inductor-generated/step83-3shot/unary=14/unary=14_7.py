
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_8 = torch.nn.ConvTranspose2d(32, 32, 3, stride=2, output_padding=(0, 2) )
    def forward(self, x1):
        v1 = self.conv_transpose_8(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 32, 12, 4)
