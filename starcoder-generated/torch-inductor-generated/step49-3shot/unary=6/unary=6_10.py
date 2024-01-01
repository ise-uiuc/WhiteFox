
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(11, stride=4, padding=2)
        self.conv = torch.nn.Conv2d(27, 19, 1, stride=1, padding=0, groups=5)
        self.bn = torch.nn.BatchNorm2d(19)
        self.act = torch.nn.ReLU()
        self.avgpool = torch.nn.AvgPool2d(3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(19, 13, 1, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.maxpool(x1)
        t2 = self.conv(t1)
        t3 = self.bn(t2)
        t4 = self.act(t3)
        t5 = self.avgpool(t4)
        t6 = self.conv2(t5)
        return t6.unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(1, 27, 224, 224)
