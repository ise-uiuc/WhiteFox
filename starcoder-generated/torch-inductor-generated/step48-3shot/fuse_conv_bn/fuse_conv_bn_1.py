
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3, affine=True, momentum=0.5)
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bias = torch.nn.Parameter(torch.empty(3))
    def forward(self, x2):
        x2 = self.conv(x2)
        x2 = self.bn(x2)
        bias = self.bias.expand_as(x)
        return x2 + bias
# Input to the model
x2 = torch.randn(1, 3, 4, 4)
