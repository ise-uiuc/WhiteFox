
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool2d = torch.nn.AvgPool2d(kernel_size=11, stride=1, padding=5)
        self.conv2d = torch.nn.Conv2d(8, 5, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.avg_pool2d(x1)
        v2 = self.conv2d(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 8, 32, 32)
