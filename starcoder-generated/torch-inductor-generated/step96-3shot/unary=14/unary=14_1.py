
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(150, 34, kernel_size=3, stride=9, padding=12)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(34, 69, kernel_size=3, stride=6, padding=6)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(69, 34, kernel_size=3, stride=12, padding=12)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(34, 97, kernel_size=3, stride=14, padding=14)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(97, 150, kernel_size=3, stride=7, padding=7)
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
        return v13
# Inputs to the model
x1 = torch.randn(2, 150, 32, 32)
