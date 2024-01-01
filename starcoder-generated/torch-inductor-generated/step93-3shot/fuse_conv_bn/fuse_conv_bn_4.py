
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(12, 6, 1, stride=1)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(6)
        torch.manual_seed(1)
        self.relu = torch.nn.ReLU()
        torch.manual_seed(1)
        self.conv_out = torch.nn.Conv2d(6, 16, 1, stride=1)
    def forward_sub1(self, x):
        x = self.conv(x)
        return x
    def forward_sub2(self, x):
        x = self.bn(x)
        return x
    def forward_sub3(self, x):
        x = self.relu(x)
        x = self.conv_out(x)
        return x
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = Model1()
    def forward(self, x1, x2):
        y1 = self.m1.forward_sub1(x1)
        y1 = self.m1.forward_sub2(y1)
        y1 = self.m1.forward_sub3(y1)
        y2 = x2
        return y1 - y2
# Inputs to the model
x = torch.randn(1, 12, 4, 4)
x1 = torch.randn(1, 12, 4, 4)
x2 = torch.randn(1, 16, 4, 4)
