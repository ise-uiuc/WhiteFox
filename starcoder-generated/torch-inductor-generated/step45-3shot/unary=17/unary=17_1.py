
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvTranspose1 = torch.nn.ConvTranspose2d(3, 3, kernel_size=3, padding=0, output_padding=0, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.ConvTranspose1(x1)
        v2 = v1 - 2.0
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
