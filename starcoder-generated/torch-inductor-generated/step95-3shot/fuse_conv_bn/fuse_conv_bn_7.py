
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, 6, name='conv')
        self.bn = torch.nn.BatchNorm2d(4, affine=False)
    def forward(self, x7):
        x7 = self.conv(x7)
        x7 = self.bn(x7)
        return x7
# Inputs to the model
x7 = torch.randn(2, 2, 6, 6)
