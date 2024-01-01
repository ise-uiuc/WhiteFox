
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(23, 43, 4, stride=2, padding=1)
        self.conv_transpose_10 = torch.nn.ConvTranspose2d(43, 52, 7, stride=2, padding=2)
        self.conv_transpose_13 = torch.nn.ConvTranspose2d(52, 7, 16, stride=2, padding=4)
        self.conv_transpose_17 = torch.nn.ConvTranspose2d(7, 1, 7, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_6(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_10(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_13(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        v10 = self.conv_transpose_17(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 23, 19, 19)
