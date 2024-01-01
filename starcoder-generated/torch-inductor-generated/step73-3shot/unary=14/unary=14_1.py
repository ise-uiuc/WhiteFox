
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_4 = torch.nn.ConvTranspose1d(4, 3, 4, stride=2, padding=0, groups=1, dilation=1, bias=True, padding_mode=False, output_padding=[0])
    def forward(self, x1):
        v1 = self.conv_transpose_4(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 8, 8)
