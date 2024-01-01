
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv= torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + v1
        v3 = torch.sigmoid(v2)
        v4 = v1 + v1
        v5 = torch.sigmoid(v4)
        return (v3 + v5)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
