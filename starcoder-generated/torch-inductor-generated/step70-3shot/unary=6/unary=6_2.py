
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=10, stride=1, padding=5, ceil_mode=True, count_include_pad=False)
    def forward(self, x1):
        h1 = self.avgpool(x1)
        h2 = h1 - 3
        h3 = torch.clamp(h2, 0, 6)
        h4 = h1 * h3
        h5 = h4 / 6
        return h5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
