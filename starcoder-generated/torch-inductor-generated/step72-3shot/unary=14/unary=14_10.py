
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(96, 40, 2, stride=2, padding=0)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(40, 48, 2, stride=1, padding=1)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(48, 32, 3, stride=1, padding=1)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(32, 30, 2, stride=1, padding=0)
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
x1 = torch.randn(32, 96, 32, 32)
