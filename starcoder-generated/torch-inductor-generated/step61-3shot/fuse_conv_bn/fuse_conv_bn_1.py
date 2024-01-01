
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 2)
        self.bn = torch.nn.ModuleList([torch.nn.BatchNorm2d(1)])
        self.conv2 = torch.nn.Conv2d(1, 1, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn[0](x)
        x = self.conv2(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
