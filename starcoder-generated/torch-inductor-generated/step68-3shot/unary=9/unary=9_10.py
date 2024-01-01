
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 1
        padding = 1
        in_channels = 3
        out_channels = 8
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 3
        t3 = torch.clamp_min(t2, 0)
        t4 = torch.clamp_max(t3, 6)
        t5 = t4 / 6
        return t5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
