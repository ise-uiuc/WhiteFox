
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 1, stride=1, padding=1)
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(8, 9, 1, stride=1, padding=1), torch.nn.BatchNorm2d(9))
    def forward(self, x1, other):
        v1 = self.conv1(x1)
        v2 = self.layer1(v1 + other)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
other = 1
