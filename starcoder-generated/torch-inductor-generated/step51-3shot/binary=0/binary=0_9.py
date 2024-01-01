
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(48, 1, 1, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(1)
        self.avgpool = torch.nn.AvgPool2d(3, 2, 1)
        self.conv2 = torch.nn.Conv2d(8, 48, 1, stride=1, padding=0)
        self.bn2 = torch.nn.BatchNorm2d(48)
    def forward(self, x1, other=None, conv3_running_var=None, conv1_weight=None, add_11=None):
        if other == None:
            other = torch.randn(x1.shape)
        v1 = self.conv1(x1)
        v1 = self.bn1(v1)
        if add_11 == None:
            add_11 = torch.randn(v1.shape)
        v1 = v1 + add_11
        v1 = self.avgpool(v1)
        v1 = self.conv2(v1)
        if conv3_running_var == None:
            conv3_running_var = torch.randn(v1.shape)
        v1 = self.bn2(v1, running_var=conv3_running_var)
        v1 = v1 + other
        return v1
# Inputs to the model
x1 = torch.randn(1, 48, 64, 64)
other = None
