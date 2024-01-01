
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(64, 256, 3, stride=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(256, 64, 1, stride=1)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(64, 32, 4, stride=5)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(32, 64, 1, stride=5)
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(64, 256, 9, stride=7)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv_transpose_2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv_transpose_3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv_transpose_4(v6)
        v8 = torch.sigmoid(v7)
        v9 = self.conv_transpose_5(v8)
        v10 = torch.sigmoid(v9)
        return v10
# Inputs to the model
x1 = torch.randn(2, 64, 4, 4)
