
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_9 = torch.nn.ConvTranspose2d(49, 16, 13, stride=1, padding=6)
        self.conv_transpose_10 = torch.nn.ConvTranspose2d(16, 27, 7, stride=1, padding=3, output_padding=1)
        self.conv_transpose_11 = torch.nn.ConvTranspose2d(27, 22, 7, stride=2, padding=3, output_padding=1)
        self.conv_transpose_12 = torch.nn.ConvTranspose2d(22, 49, 7, stride=1, padding=3, output_padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_9(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_10(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_11(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        v10 = self.conv_transpose_12(v9)
        v11 = torch.sigmoid(v10)
        v12 = v10 * v11
        return v12
# Inputs to the model
x1 = torch.randn(1, 49, 8, 8)
