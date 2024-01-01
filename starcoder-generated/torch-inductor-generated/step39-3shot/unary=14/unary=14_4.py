
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
        self.conv_transpose_7 = torch.nn.ConvTranspose2d(512, 256, 4, stride=1, padding=1)
        self.conv_transpose_8 = torch.nn.ConvTranspose2d(256, 128, 4, stride=1, padding=1)
        self.conv_transpose_9 = torch.nn.ConvTranspose2d(128, 64, 4, stride=1, padding=1)
        self.conv_transpose_10 = torch.nn.ConvTranspose2d(63, 3, 4, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_6(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_7(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_8(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        v10 = self.conv_transpose_9(v9)
        v11 = torch.sigmoid(v10)
        v12 = v10 * v11
        v21 = torch.cat((v12, x1), 1)
        v13 = self.conv_transpose_10(v21)
        v14 = torch.sigmoid(v13)
        v15 = v13 * v14
        return v15
# Inputs to the model
x1 = torch.randn(1, 512, 8, 8)
