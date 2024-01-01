
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1, stride=1, padding=0, bias=False)
        self.batch_norm = nn.BatchNorm2d(1, momentum=0.1, eps=1.1e-05, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(1, 1, 1, stride=1, padding=0, bias=False)

    def forward(self, x3):
        if (x3 is not None):
           opt_conv_0 = self.conv1(x3)
        opt_batch_norm_0 = self.batch_norm(opt_conv_0)
        opt_conv_1 = self.conv2(opt_batch_norm_0)
        return opt_conv_1
# Inputs to the model
x3 = torch.randn(1, 1, 4, 4)
