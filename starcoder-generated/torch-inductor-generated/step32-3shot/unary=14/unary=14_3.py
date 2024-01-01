
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose3d(8, 13, 4, stride=(1, 1, 1), padding=(2, 1, 0))
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 5, 10, 20)
