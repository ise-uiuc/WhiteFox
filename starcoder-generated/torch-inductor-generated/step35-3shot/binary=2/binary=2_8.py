
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(5, 1, kernel_size=2, stride=2, padding=0, output_padding=0, groups=1, dilation=1, bias=True)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - None
        return v2
# Inputs to the model
x = torch.randn(1, 5, 64, 64)
