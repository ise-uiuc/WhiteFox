
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.relu2 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.bn(y)
        return y
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
