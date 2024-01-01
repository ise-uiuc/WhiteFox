
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_15 = torch.nn.ConvTranspose2d(128, 16, 3, stride=2, padding=1, output_padding=1)
        self.conv_transpose_17 = torch.nn.ConvTranspose2d(512, 65, 3, stride=2, padding=1, output_padding=1)
        self.conv_transpose_20 = torch.nn.ConvTranspose2d(256, 65, 4, stride=2, padding=1, output_padding=1)
        self.conv_transpose_24 = torch.nn.ConvTranspose2d(64, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_15(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_17(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = v1 + v6
        v8 = self.conv_transpose_20(v7)
        v9 = torch.sigmoid(v8)
        v10 = v8 * v9
        v11 = v3 + v10
        v12 = self.conv_transpose_24(v11)
        v13 = torch.sigmoid(v12)
        v14 = v12 * v13
        return v14
# Inputs to the model
x1 = torch.randn(1, 128, 19, 19)
