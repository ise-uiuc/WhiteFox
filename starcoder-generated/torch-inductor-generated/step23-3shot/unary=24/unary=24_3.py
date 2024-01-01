
class Model(torch.nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding)

    def forward(self, x, negative_slope):
        v1 = self.conv1(x)
        v3 = v1 > 0
        v4 = v1 * negative_slope
        v5 = torch.where(v3, v1, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 8, 256, 256)
x2 = 0
