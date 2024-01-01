
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_10 = torch.nn.ConvTranspose2d(300, 300, 3, stride=(1, 1), padding=(2, 2), dilation=2)
    def forward(self, x1):
        v1 = self.conv_transpose_10(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 300, 151, 151)
