
class ConvTranspose3dModel(torch.nn.Module):
    def __init__(self):
        super(ConvTranspose3dModel, self).__init__()
        self.conv_transpose_layer = torch.nn.ConvTranspose3d(43, 47, 4, stride=1, padding=1, output_padding=0, groups=1, bias=True, dilation=2)
    def forward(self, x1):
        x2 = self.conv_transpose_layer(x1)
        x3 = x2 > 0
        x4 = x2 * 0.4
        x5 = torch.where((x3), (x2), (x4))
        return x5
# Inputs to the model
x6 = torch.randn(1, 43, 7, 11, 14)
