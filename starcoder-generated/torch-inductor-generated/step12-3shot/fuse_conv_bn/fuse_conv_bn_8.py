
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(4, 8, 3)
        self.bn0 = torch.nn.BatchNorm2d(4)
    def forward(self, x3):
        bn0 = self.bn0(x3)
        return self.conv0(bn0)
# Inputs to the model
x3 = torch.randn(1, 8, 8, 4)
