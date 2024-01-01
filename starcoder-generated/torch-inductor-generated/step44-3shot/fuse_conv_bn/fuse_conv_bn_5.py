
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(7, 10, 1, 1, 0, 1, 1, bias=False)
        self.bn = torch.nn.BatchNorm2d(10, affine=False)
    def forward(self, input):
        conv = self.conv1(input)
        bn = self.bn(self.conv1(input))
        return bn
# Inputs to the model
input = torch.randn(1, 7, 3, 3)
