
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        padding = 1
        kernel_size = 4
        stride = 3
        dilation = 2
        input_channel = 3
        output_channel = 2
        padding_h = 2 * padding - stride + kernel_size
        padding_w = 4 * padding - stride + kernel_size
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 128, (3, padding_w), (3, stride), (2, dilation))
        self.conv_transpose.groups = input_channel
        self.conv_transpose.out_channels = output_channel
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 128, 20, 20)
