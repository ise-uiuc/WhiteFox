
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, input):
        t1 = self.conv(input)
        t2 = t1.view(3, 9)
        t3 = self.bn(t2)
        return t3.view(1, 9, 3, 3)
# Inputs to the model
input = torch.randn(1, 3, 224, 224)
