
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
        self.avgpool = torch.nn.AvgPool2d(7, stride=1)
 
    def forward(self, x1):
        r = self.conv(x1)
        x = self.bn(r)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
