
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = M.Conv2d(kernel_size=3, in_channels=3, offset_groups=1, padding=1, dilation=2)
        self.batch_norm = M.BatchNorm2d(num_features=1, affine=1, eps=1e-05, momentum=0.999, track_running_stats=False)
        self.conv2 = M.Conv2d(kernel_size=3, in_channels=1, offset_groups=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 + v1
        v3 = self.batch_norm(v2)
        v4 = self.conv2(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
