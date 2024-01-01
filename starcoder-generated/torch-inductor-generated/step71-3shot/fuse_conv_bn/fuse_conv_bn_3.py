
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 2, (2, 3, 3), stride=(2, 2, 1), padding=(3, 1, 2), dilation=(2, 1, 2), groups=1)
        self.bn = torch.nn.BatchNorm3d(2, affine=False, track_running_stats=True)
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(x)
        return y
# Inputs to the model
x = torch.randn(1, 1, 4, 4, 4)
