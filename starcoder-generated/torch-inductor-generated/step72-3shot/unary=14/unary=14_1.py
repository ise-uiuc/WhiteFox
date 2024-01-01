
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(1783, 201, 1, stride=1, padding=0)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(201, 201, 1, stride=1, padding=0)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(201, 180, 5, stride=1, padding=2, dilation=1)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(180, 168, 5, stride=1, padding=2, dilation=1)
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(168, 154, 7, stride=1, padding=3, dilation=1)
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(154, 138, 7, stride=1, padding=3)
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
        v13 = self.conv_transpose_5(v12)
        v14 = torch.sigmoid(v13)
        v15 = v13 * v14
        v16 = self.conv_transpose_6(v15)
        v17 = torch.sigmoid(v16)
        v18 = v16 * v17
        return v18
# Inputs to the model
x1 = torch.randn(1, 1783, 32, 32)
