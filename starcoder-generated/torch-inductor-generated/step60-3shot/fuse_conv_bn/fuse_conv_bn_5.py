
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(1, 3, 3, padding=1, bias=False)
        self.conv1 = torch.nn.Conv2d(3, 1, 1, bias=False)
        self.bn = torch.nn.BatchNorm2d(3, momentum=0.1)
    def forward(self, x2):
        x = self.conv0(x2)
        x = self.conv1(F.relu(self.bn(x)))
        return x2 + x
# Inputs to the model
x2 = torch.randn(1, 1, 4, 4)
