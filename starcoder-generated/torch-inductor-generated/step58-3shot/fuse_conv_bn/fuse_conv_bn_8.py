
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(2, 2))
        self.bn = torch.nn.BatchNorm2d(num_features=1, affine=False)
    def forward(self, x1):
        x1 = self.bn(x1)
        y = self.conv(x1)
        return y
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)
