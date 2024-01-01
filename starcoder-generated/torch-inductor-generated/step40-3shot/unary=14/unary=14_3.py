
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(189, 201, 5, stride=1, padding=1)
        self.conv_transpose_7 = torch.nn.ConvTranspose2d(201, 29, 1, stride=1, padding=0)
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(29, 1, 1, stride=1, padding=0)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(189, 11, 2, stride=1, padding=1)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(11, 1, 2, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_5(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_7(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_0(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        v10 = self.conv_transpose_3(x1)
        v11 = torch.sigmoid(v10)
        v12 = v10 * v11
        v13 = self.conv_transpose_4(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 189, 13, 13)
