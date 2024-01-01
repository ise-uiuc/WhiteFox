
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        torch.manual_seed(2)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x2):
        y2 = self.conv1(x2)
        y3 = self.bn(y2)
        return y3
# Inputs to the model
x2 = torch.randn(1, 3, 10, 10)
