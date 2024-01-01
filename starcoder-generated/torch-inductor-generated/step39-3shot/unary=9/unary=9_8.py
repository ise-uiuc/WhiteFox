
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(8, momentum=0.5)
    def forward(self, x1):
        t1 = self.relu(x1)
        t2 = self.conv(t1)
        t3 = self.bn(t2)
        return t3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
