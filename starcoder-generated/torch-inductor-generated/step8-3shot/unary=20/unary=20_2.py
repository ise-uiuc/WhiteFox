
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(1, out_channels=1, kernel_size=5, stride=1, padding=1, output_padding=1, groups=1, bias=True, dilation=1)

    def forward(self, x1):
        v1 = self.conv_t(x1)
        return v1

# Inputs to the model
x1 = torch.randn(1, 1, 11)
