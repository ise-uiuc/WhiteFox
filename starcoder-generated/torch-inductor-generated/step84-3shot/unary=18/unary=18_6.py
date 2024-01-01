
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=3, out_channels=64, kernel_size=10, stride=3, padding=0, output_padding=0, groups=1, dilation=1, bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels=432, out_channels=128, kernel_size=2, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.deconv1(x1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 45, 45)
