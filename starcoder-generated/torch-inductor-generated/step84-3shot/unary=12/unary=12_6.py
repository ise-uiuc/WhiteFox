
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=(100, 3), bias=False)
        self.bn = torch.nn.BatchNorm2d(1, affine=True)
        self.act = torch.nn.Sigmoid()
    def forward(self, x1):  # pylint: disable=arguments-differ
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = self.act(v2)
        v4 = v2 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 800, 8)
