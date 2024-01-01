
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
