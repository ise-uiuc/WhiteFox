
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv_transpose_1(v1)
        v3 = torch.sigmoid(v2)
        v4 = v2 * v3
        v5 = self.conv_transpose(v4)
        v6 = self.conv_transpose_1(v3)
        v7 = torch.sigmoid(v4)
        v8 = v4 * v7
        v9 = v5 + v8
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
