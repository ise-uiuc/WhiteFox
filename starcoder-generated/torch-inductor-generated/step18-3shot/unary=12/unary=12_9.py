
kernel_size = 3
strides = 1
padding = (kernel_size - 1) // 2
conv_module = torch.nn.Conv2d(
            in_channels=5,
            out_channels=5,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = conv_module    
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
