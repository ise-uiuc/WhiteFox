
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_100 = torch.nn.ConvTranspose3d(4, 64, (3, 2, 4), stride=(3, 2, 5), padding=(1, 0, 0), output_padding=(2, 1, 1))
    def forward(self, x1):
        v1 = self.conv_transpose_100(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 104, 128, 64)
