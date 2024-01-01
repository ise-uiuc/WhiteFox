
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(4, 42, 6, stride=2, padding=2, dilation=1)
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(140, 57, 7, stride=2, padding=3, dilation=1)
        self.conv_transpose_8 = torch.nn.ConvTranspose2d(57, 4, 7, stride=2, padding=3, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose_3(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_6(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_8(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 4, 15, 15)
