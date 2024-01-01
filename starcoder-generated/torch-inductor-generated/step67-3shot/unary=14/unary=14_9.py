
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_18 = torch.nn.ConvTranspose2d(93, 197, 9, stride=2, padding=4, dilation=1)
        self.conv_transpose_20 = torch.nn.ConvTranspose2d(162, 414, 8, stride=2, padding=3, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose_18(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_20(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 93, 16, 16)
