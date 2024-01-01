
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 2, 1)
        self.bn = torch.nn.BatchNorm2d(2)
# Input to the model
    def forward(self, input):
        t1 = self.bn(self.conv1(input))
        t2 = self.bn(t1)
        return t2
# Input to the model
input = torch.randn(1, 3, 5, 5)
