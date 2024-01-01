
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 6)
        v4 = v1 * v3
        v5 = v4 / 6
        v6 = self.maxpool(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
