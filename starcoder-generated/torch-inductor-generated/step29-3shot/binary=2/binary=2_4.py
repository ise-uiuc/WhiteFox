
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 0.1
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
