 (with a typo)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, (4, 3), stride=(2, 2), dilation=4, groups=1, bias=True, padding=(2, 1), padding_mode='zeros')
        self.bn = torch.nn.BatchNorm2d(1, affine=True, track_running_stats=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 64, 128)
