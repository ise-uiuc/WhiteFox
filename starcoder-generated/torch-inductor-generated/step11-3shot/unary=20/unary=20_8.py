
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=20, stride=15, padding=10, dilation=1, groups=1, bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 300)
