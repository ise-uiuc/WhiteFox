
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, (1, 9), stride=(1, 1), padding=(1, 0))
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.pool1 = torch.nn.MaxPool2d((1, 5), stride=1, padding=(1, 2))
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.conv2 = torch.nn.Conv2d(3, 64, (1, 1), stride=(1, 1), padding=0)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = self.bn1(t1)
        t3 = self.pool1(t2)
        t4 = self.avgpool(x1)
        t5 = self.conv2(t4)
        t6 = t3.add(t5)
        return t6
# Inputs to the model
x1 = torch.randn(1, 3, 1024, 7)
