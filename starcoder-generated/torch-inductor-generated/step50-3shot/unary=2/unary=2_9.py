
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(33, 83, 8, stride=2, padding=2, groups=1, bias=True, dilation=1)
        self.batch_norm2d = torch.nn.BatchNorm2d(142, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.batch_norm2d(v1)
        v3 = self.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 33, 51, 91)
