
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 4, 3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.relu = torch.nn.ReLU(inplace=False)
        self.avg_pool = torch.nn.AvgPool2d(6, stride=1, padding=1)
    def forward(self, input, affine=True):
        t0 = self.conv(input)
        t1 = self.bn(t0)
        t2 = self.relu(t1)
        t3 = self.avg_pool(t2)
        return t3
# Inputs to the model
input = torch.randn(1, 10, 28, 28)
affine = True
