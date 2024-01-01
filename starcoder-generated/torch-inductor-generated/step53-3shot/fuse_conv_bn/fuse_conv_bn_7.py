
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 3, 3, 1, 1, 1)
        self.bn1 = torch.nn.BatchNorm2d(2)
        self.conv2 = torch.nn.Conv2d(1, 16, 7)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(1, 1, 1)
        self.bn3 = torch.nn.BatchNorm2d(1, affine=True)
    def forward(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.bn1(x1)
        x3 = self.conv2(x2)
        x4 = self.bn2(x3)
        x5 = self.conv3(x4)
        x6 = self.bn3(x5)
        return x6

class Model2(torch.nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 3, 3, 1, 1, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(2, affine=True)
        self.conv2 = torch.nn.Conv2d(1, 16, 7, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(16, affine=True)
        self.conv3 = torch.nn.Conv2d(1, 1, 1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(1)
    def forward(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.bn1(x1)
        x3 = self.conv2(x2)
        x4 = self.bn2(x3)
        x5 = self.conv3(x4)
        x6 = self.bn3(x5)
        return x6

# Inputs to the model
inputs = torch.randn(1, 2, 224, 224)
