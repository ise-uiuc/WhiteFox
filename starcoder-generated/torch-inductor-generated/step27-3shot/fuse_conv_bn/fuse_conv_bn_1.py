
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(3, 3, 3)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(1)
        self.bn.weight.data = torch.ones((1))
        self.bn.bias.data = torch.zeros((1))
    def forward(self, x2):
        v1 = self.bn(x2)
        v2 = self.conv(v1)
        v2 = self.bn(v2)
        return v2
# Inputs to the model
x2 = torch.ones(1, 3, 2, 2) * 10000000
