
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(15, 13, (4, 4), 1, 0)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(13, 8, (4, 4), 1, 0)
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(8, 5, (4, 4), 1, 0)
    def forward(self, x1):
        v1 = self.conv_transpose_3(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_4(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_5(v6)
        return v7
# Inputs to the model
x1 = torch.randn(2, 15, 32, 32)
