
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1024, 4096, 1, stride=2, padding=0, dilation=1, groups=1)
        self.transpose_conv = torch.nn.ConvTranspose2d(4096, 4096, 5, stride=2, padding=0, output_padding=0, groups=1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.transpose_conv(v1)
        v3 = self.sigmoid(v2)
        v4 = v1 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 1024, 2, 2)
