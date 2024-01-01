
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(244, 654, 3, stride=1, padding=4, dilation=2)
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(654, 324, 1, stride=1, padding=1, dilation=1)
        self.conv_transpose_7 = torch.nn.ConvTranspose2d(324, 132, 3, stride=1, padding=1, dilation=1)
        self.conv_transpose_9 = torch.nn.ConvTranspose2d(132, 76, 1, stride=1, padding=1, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose_3(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_5(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_7(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        v10 = self.conv_transpose_9(v9)
        v11 = torch.sigmoid(v10)
        v12 = v10 * v11
        return v12
# Inputs to the model
x1 = torch.randn(1, 244, 64, 64)
