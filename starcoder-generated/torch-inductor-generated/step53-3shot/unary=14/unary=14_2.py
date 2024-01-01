
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_7 = torch.nn.ConvTranspose3d(2, 40, 6, stride=3, padding=2)
        self.conv_transpose_9 = torch.nn.ConvTranspose3d(40, 20, 7, stride=2, padding=2)
        self.conv_transpose_11 = torch.nn.ConvTranspose3d(20, 5, 12, stride=6, padding=4)
        self.conv_transpose_13 = torch.nn.ConvTranspose3d(5, 1, 2, stride=3, padding=2)
    def forward(self, x1):
        v1 = self.conv_transpose_7(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_9(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_11(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        v10 = self.conv_transpose_13(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 2, 8, 8, 8)
