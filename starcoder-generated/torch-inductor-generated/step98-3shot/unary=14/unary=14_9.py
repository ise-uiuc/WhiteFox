
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose3d(7, 7, 7, stride=1, padding=3)
        self.conv_transpose_2 = torch.nn.ConvTranspose3d(7, 5, 9, stride=1, padding=4)
        self.conv_transpose_3 = torch.nn.ConvTranspose3d(5, 10, 11, stride=1, padding=5)
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
x1 = torch.randn(1, 7, 16, 16, 16)
