
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose3d(1, 25, 5, stride=2, padding=0, output_padding=3, dilation=3)
        self.conv_transpose_2 = torch.nn.ConvTranspose3d(25, 50, 5, stride=2, padding=2, output_padding=0, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_2(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28, 28)
