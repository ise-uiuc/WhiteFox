
class Model1(nn.Module):
    def __init__(self, channels=8, padding=1):
        super().__init__()
        self.conv_b1 = (
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=1, stride=1, groups=8, bias=True))
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=padding)
        self.conv_a1 = (
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, groups=8, bias=True))
    def forward(self, data):
        x1 = self.conv_b1(data)
        out = self.conv(x1)
        out = self.conv_a1(out)
        return out

class Model1(nn.Module):
    def __init__(self, channels=8):
        super().__init__()
        self.conv_b1 = (
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=1, stride=1, groups=8, bias=True))
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1,padding=1)
        self.conv_a1 = (
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, groups=8, bias=True))
    def forward(self, data):
        x1 = self.conv_b1(data)
        v1 = self.conv(x1)
        v2 = self.conv_a1(v1)
        v3 = v1.add(v2)
        v4 = v3.add(3)
        v5 = v4.clamp(0, 6)
        v6 = v5.div(6)
        return v6
# Inputs to the model
x1 = torch.randn(5, 3, 224, 224)
