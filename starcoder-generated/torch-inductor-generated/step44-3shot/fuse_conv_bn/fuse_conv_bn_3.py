
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(5, affine=False)
    def forward(self, x1):
        y1 = self.conv1(x1)
        y2 = self.bn1(y1)
        y3 = self.conv1(y2)
        return y3
# Inputs to the model
x1 = torch.randn(1, 3, 5, 5)
