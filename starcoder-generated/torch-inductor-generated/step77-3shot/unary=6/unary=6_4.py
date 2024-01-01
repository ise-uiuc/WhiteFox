
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool_2d = torch.nn.Sequential(torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False))

    def forward(self, x):
        t0 = torch.clamp_max(x, 6)
        return self.avg_pool_2d(t0)
# Inputs to the model
x= torch.randn(1, 1)
