
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_7 = torch.nn.ConvTranspose1d(48, 7, 5, stride=2, padding=3, bias=True, dilation=1, groups=1)
    def forward(self, x1):
        v1 = self.conv_transpose_7(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 48, 2048)
