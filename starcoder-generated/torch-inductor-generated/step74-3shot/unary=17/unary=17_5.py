
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 11, 3, padding=1, dilation=1, groups=1)
        # Pointwise transposed convolution
        # This operation produces `n` output values for each `m` channel values from `input` tensor.
        # The number of input channels and the number of output channels may be different.
        self.conv_transpose = torch.nn.ConvTranspose2d(11, 21, 1, dilation=1, stride=1, padding=0, output_padding=0, groups=1, bias=True, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.conv_transpose(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 35, 35)
