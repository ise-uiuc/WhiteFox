
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(2, 5, 1, stride=1, padding=0)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(2, 5, (1, 1), stride=1, padding=0)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(2, 5, (3, 3), stride=1, padding=0)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(2, 5, (3, 3), stride=1, padding=1)
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(2, 5, (3, 3), stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv_transpose_2(x1)
        v3 = self.conv_transpose_3(x1)
        v4 = self.conv_transpose_4(x1)
        v5 = self.conv_transpose_5(x1)
        v6 = torch.sigmoid(v1)
        v7 = v1 * v6
        v8 = torch.sigmoid(v2)
        v9 = v2 * v8
        v10 = torch.sigmoid(v3)
        v11 = v3 * v10
        v12 = torch.sigmoid(v4)
        v13 = v4 * v12
        v14 = torch.sigmoid(v5)
        v15 = v5 * v14
        return v1, v2, v3, v4, v5
# Inputs to the model
x1 = torch.randn(1, 2, 128, 128)
