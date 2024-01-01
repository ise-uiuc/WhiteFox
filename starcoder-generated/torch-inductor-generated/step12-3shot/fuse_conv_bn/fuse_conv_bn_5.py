
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(4, 8, 3)
        self.bn0 = torch.nn.BatchNorm2d(8)
    def forward(self, x2):
        return self.conv0(self.bn0(x2))
# Inputs to the model
x2 = torch.randn(1, 4, 4, 4)
