
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv1 = torch.nn.Conv1d(3, 3, 2, stride=2, padding=0, bias=True)
        torch.manual_seed(1)
        self.bn_1 = torch.nn.BatchNorm1d(3, momentum=0.9, affine=False)
        torch.manual_seed(1)
        self.conv2 = torch.nn.Conv2d(1, 3, 1, stride=2, padding=0, bias=True)
        torch.manual_seed(1)
        self.bn2 = torch.nn.BatchNorm2d(1)
        torch.manual_seed(1)
        self.conv3 = torch.nn.Conv2d(1, 1, 0, stride=2, padding=0, bias=True)
        torch.manual_seed(1)
        self.bn3 = torch.nn.BatchNorm2d(1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn_1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 10, 10)

# Model begins
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight0 = torch.nn.Parameter(torch.randn(8, 8, 8, 8))
        self.weight1 = torch.nn.Parameter(torch.randn(8, 8, 8, 8))
        self.weight2 = torch.nn.Parameter(torch.randn(8, 8, 8, 8))
        self.weight3 = torch.nn.Parameter(torch.randn(8, 8, 8, 8))
        self.weight4 = torch.nn.Parameter(torch.randn(8, 8, 8, 8))
        self.weight5 = torch.nn.Parameter(torch.randn(8, 8, 8, 8))
        self.weight6 = torch.nn.Parameter(torch.randn(8, 8, 8, 8))

    def forward(self, x):
        weight0 = self.weight0
        weight = weight0
        weight1 = self.weight1
        weight = weight + weight1
        weight2 = self.weight2
        weight = weight + weight2
        weight3 = self.weight3
        weight = weight + weight3
        weight4 = self.weight4
        weight = weight + weight4
        weight5 = self.weight5
        weight = weight + weight5
        weight6 = self.weight6
        weight = weight + weight6
        return x
# Inputs to the model
x = torch.rand((1, 8, 10, 10))
