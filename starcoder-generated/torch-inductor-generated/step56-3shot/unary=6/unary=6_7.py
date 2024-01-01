
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 13, 2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.avgpool = torch.nn.AvgPool2d(2, stride=2, padding=0)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = self.relu(t1 + 3)
        t3 = self.avgpool(t2)
        return t3
# Inputs to the model
x1 = torch.randn(2, 3, 28, 28)
