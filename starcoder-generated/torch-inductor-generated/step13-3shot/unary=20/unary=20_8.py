
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(999, in_channels=4, out_channels=4, kernel_size=3, stride=3, padding=34, dilation=34, groups=88)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(245, 999, 5, 5)
