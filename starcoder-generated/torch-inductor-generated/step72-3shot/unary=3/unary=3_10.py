
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d()
        self.conv1.in_channels = 2
        self.conv1.out_channels = 3
        self.conv1.kernel_size = (1)
        self.conv2 = torch.nn.Conv2d()
        self.conv2.in_channels = 3
        self.conv2.out_channels = 2
        self.conv2.kernel_size = (1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)
