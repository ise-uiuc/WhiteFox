
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=2)
        self.bn = torch.nn.BatchNorm2d(3, affine=False)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 10, 10)
