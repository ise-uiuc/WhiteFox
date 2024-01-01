
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_11 = torch.nn.ConvTranspose2d(128, 99, 8, stride=2, padding=7, dilation=2)
        self.conv_transpose_13 = torch.nn.ConvTranspose2d(99, 60, 5, stride=1, padding=1, dilation=1)
        self.conv_transpose_15 = torch.nn.ConvTranspose2d(60, 32, 11, stride=1, padding=4, dilation=2)
        self.conv_transpose_17 = torch.nn.ConvTranspose2d(32, 16, 1, stride=1, padding=1, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose_11(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_13(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_15(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        v10 = self.conv_transpose_17(v9)
        v11 = torch.sigmoid(v10)
        v12 = v10 * v11
        return v12
# Inputs to the model
x1 = torch.randn(1, 128, 64, 64)
