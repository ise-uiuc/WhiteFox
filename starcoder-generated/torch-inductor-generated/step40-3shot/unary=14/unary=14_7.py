
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_8 = torch.nn.ConvTranspose2d(2, 40, 4, stride=2, padding=1)
        self.conv_transpose_10 = torch.nn.ConvTranspose2d(40, 20, 7, stride=2, padding=2)
        self.conv_transpose_12 = torch.nn.ConvTranspose2d(20, 8, 16, stride=2, padding=4)
        self.conv_transpose_14 = torch.nn.ConvTranspose2d(8, 1, 7, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_8(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_10(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_12(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        v10 = self.conv_transpose_14(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 2, 14, 14)
