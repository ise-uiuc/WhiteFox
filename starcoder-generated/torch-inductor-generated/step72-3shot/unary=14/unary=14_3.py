
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(1613, 3, 7, stride=2, padding=1, dilation=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(3, 401, 1, stride=1, padding=0)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(401, 1, 7, stride=1, padding=3)
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
        return v9
# Inputs to the model
x1 = torch.randn(1, 1613, 109, 109)
