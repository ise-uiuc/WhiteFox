
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        conv1 = torch.nn.Conv2d(32,
                               out_channels=64,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=1,
                               bias=True)
        pointwise = torch.nn.Conv2d(64,
                                    out_channels=64,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    dilation=1,
                                    groups=1,
                                    bias=True)
        bn = torch.nn.BatchNorm2d(64,
                                  running_mean=None,
                                  running_var=None,
                                  momentum=0.1,
                                  eps=1e-05,
                                  track_running_stats=False)
        self.block1 = torch.nn.Sequential(conv1, bn, pointwise)
    def forward(self, x):
        x = self.block1(x)
        return x
# Inputs to the model
x1 = torch.randn(10, 32, 10, 10)
