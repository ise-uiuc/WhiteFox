
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(11, 7, 8, stride=1, padding=2, dilation=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(7, 1, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_2(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 11, 8, 8)
