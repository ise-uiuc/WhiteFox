
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn2 = torch.nn.BatchNorm1d(3)
        self.avgpool = torch.nn.AvgPool2d(3, stride=1, padding=1)
    def forward(self, x1):
        # NOTE: Here we use F.relu instead of torch.relu
        t1 = self.conv(x1)
        t2 = self.bn1(t1)
        # NOTE: Here we replace torch.relu with torch.nn.functional.relu
        t3 = F.relu(3 + t2)
        t4 = torch.clamp(t3, 0, 6)
        t5 = t1 * t4
        t6 = t5 / 6
        t7 = self.bn2(t6)
        t8 = torch.relu(t7)
        t9 = self.avgpool(t8)
        return t9
# Inputs to the model
x1 = torch.randn(2, 3, 15, 15)
