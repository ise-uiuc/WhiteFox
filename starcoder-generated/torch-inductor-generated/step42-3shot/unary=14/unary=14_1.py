
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(11, 143, 1, stride=1, padding=0)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(143, 256, 3, stride=1, padding=0, groups=1, dilation=2)
    def forward(self, x1):
        v1 = self.conv_transpose_0(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_1(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 11, 321, 321)
