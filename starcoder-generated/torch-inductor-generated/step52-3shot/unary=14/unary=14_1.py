
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_18 = torch.nn.ConvTranspose2d(2, 5, 7)
        self.conv_transpose_20 = torch.nn.ConvTranspose2d(5, 6, 15)
    def forward(self, x1):
        v1 = self.conv_transpose_18(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_20(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 22, 22)
