
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(6, 8, 4, stride=2, padding=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(8, 10, 4, stride=2, padding=1)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(10, 12, 4, stride=2, padding=1)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(12, 1, 2, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_2(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_3(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        v10 = self.conv_transpose_4(v9)
        v11 = torch.sigmoid(v10)
        v12 = v10 * v11
        return v12
# Inputs to the model
x1 = torch.randn(1, 6, 8, 8)
