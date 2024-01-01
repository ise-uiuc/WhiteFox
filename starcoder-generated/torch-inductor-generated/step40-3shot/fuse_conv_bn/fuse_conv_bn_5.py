
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers1 = torch.nn.Sequential(torch.nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=1, bias=False), torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False))
        bn_last = torch.nn.BatchNorm2d(64)
        self.features1 = torch.nn.Sequential()
        self.features1.add_module('layers0', layers1)
        self.features1.add_module('ReLU1', torch.nn.ReLU(inplace=False))
        self.features1.add_module('last', bn_last)
    def forward(self, x):
        output = self.features1(x)
        return output
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
