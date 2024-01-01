
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(191, 113, 3, stride=1)
        self.conv_transpose_19 = torch.nn.ConvTranspose2d(113, 22, 8, stride=2, padding=4)
        self.conv_transpose_31 = torch.nn.ConvTranspose2d(22, 9, 6, stride=1)
        self.conv_transpose_43 = torch.nn.ConvTranspose2d(9, 1, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose_6(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_19(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_31(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        v10 = self.conv_transpose_43(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 191, 35, 35)
