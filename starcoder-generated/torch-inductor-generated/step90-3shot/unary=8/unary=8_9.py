
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool_2d = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
    def forward(self, x1):
        v1 = self.avg_pool_2d(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
