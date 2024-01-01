
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 1, kernel_size=7, stride=1, padding=3, dilation=1, groups=1, bias=False)
        self.conv1 = torch.nn.ConvTranspose2d(1, 1, kernel_size=9, stride=4, padding=4, output_padding=0, groups=1, bias=True, dilation=1)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 6, 125, 100)
