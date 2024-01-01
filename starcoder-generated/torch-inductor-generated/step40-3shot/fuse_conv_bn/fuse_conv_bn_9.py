
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=12, bias=False)
        self.bn1_1 = torch.nn.BatchNorm2d(64)
        self.conv1_2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=12, bias=False)
        self.bn1_2 = torch.nn.BatchNorm2d(64)
        pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        relu1 = torch.nn.ReLU(inplace=False)
        layers1 = torch.nn.Sequential(self.conv1_1, self.bn1_1, relu1, self.conv1_2, self.bn1_2, pool1)
        self.features1 = torch.nn.Sequential()
        self.features1.add_module('layers0', layers1)
        self.features1.add_module('ReLU1', torch.nn.ReLU(inplace=False))
    def forward(self, x):
        output = self.features1(x)
        return output
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
