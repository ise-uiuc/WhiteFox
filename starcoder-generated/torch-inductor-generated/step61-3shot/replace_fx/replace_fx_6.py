
class test_module1(torch.nn.Module):
    def __init__(self):
        super(test_module1, self).__init__()
        self.block1 = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 1, stride=1, padding=0, dilation=1, out_channels=16, groups=1, bias=True),
                                          torch.nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                          torch.nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.block1(x)
        x1 = torch.nn.functional.dropout(x, p=0.01, training=True)
        x2 = torch.nn.functional.softmax(x1, dim=1)
        return x2
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
