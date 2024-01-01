
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_7 = torch.nn.ConvTranspose2d(232, 1000, 8, groups=4, dilation=1, padding=4)
    def forward(self, x1):
        v1 = self.conv_transpose_7(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(4, 232, 100, 100)
