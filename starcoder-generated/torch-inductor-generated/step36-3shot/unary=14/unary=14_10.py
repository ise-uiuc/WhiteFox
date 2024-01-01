
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(21, 14984, 3, stride=1, padding=1)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(14984, 388, 3, stride=2, padding=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(388, 3107, 3, stride=2, padding=1)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(3107, 24503, 3, stride=2, padding=1)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(24503, 50218, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_0(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_1(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_2(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        v10 = self.conv_transpose_3(v9)
        v11 = torch.sigmoid(v10)
        v12 = v10 * v11
        v13 = self.conv_transpose_4(v12)
        v14 = torch.sigmoid(v13)
        v15 = v13 * v14
        return v15
# Inputs to the model
x1 = torch.randn(1, 21, 286, 286)
