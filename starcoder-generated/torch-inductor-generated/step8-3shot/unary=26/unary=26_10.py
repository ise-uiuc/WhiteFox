
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
            dilation=1,
        )
    def forward(self, x):
        x1 = self.conv_transpose2d(x)
        x2 = x1 > 0
        x3 = x1 * 0.4
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x = torch.randn(6, 1, 224, 224)
