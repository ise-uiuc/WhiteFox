
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True)
    def forward(self, x1, x2):
        v1 = self.conv_transpose(x1)
        v2 = torch.sigmoid(v1)
        v3 = v2 - x2
        return v3
# Inputs to the model
x1 = torch.randn(100, 1, 10)
x2 = torch.randn(100, 1, 10)
