
class Module_18(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_21 = torch.nn.ConvTranspose2d(1, 54, 77, stride=2, padding=38, output_padding=1)
        self.conv_6 = torch.nn.Conv2d(54, 2, 7, stride=2, padding=3, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose_21(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_6(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 179, 179)
