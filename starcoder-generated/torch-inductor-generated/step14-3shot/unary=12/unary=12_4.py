
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1)
        self.pool = torch.nn.AvgPool2d(kernel_size=(4, 4), stride=2, padding=(1, 1), count_include_pad=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.pool(v1)
        v3 = v2.repeat(3, 1, 1, 1)
        v4 = torch.flatten(v3, 1)
        v5 = v4.sigmoid()
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
