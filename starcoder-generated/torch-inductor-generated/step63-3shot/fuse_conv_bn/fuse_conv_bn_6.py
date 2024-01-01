
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.bn = torch.nn.BatchNorm2d(1, affine=False)
    def forward(self, input):
        return self.conv(input) + self.bn(self.conv(input))
# Inputs to the model
input = torch.randn(1, 1, 1, 1)
