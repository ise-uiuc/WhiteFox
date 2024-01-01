
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_10 = torch.nn.ConvTranspose2d(250, 512, 5, stride=2, padding=1)
        self.conv_transpose_12 = torch.nn.ConvTranspose2d(512, 256, 3, stride=1, padding=0)
        self.conv_transpose_14 = torch.nn.ConvTranspose2d(256, 128, 3, stride=2, padding=0)
        self.conv_transpose_16 = torch.nn.ConvTranspose2d(128, 64, 4, stride=1, padding=0)
        self.conv_transpose_18 = torch.nn.ConvTranspose2d(64, 3, 4, stride=2, padding=3)
    def forward(self, x1):
        v1 = self.conv_transpose_10(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_12(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_14(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        v10 = self.conv_transpose_16(v9)
        v11 = torch.sigmoid(v10)
        v12 = v10 * v11
        v13 = self.conv_transpose_18(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 250, 31, 31)
