
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 5, 2)
        self.bn = torch.nn.BatchNorm2d(5)
        self.relu1 = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(5, 5, 2)
        self.relu2 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(5, 5, 2)
        self.relu3 = torch.nn.ReLU()
    def forward(self, x):
        x = self.relu1(x)
        x = self.conv(x)
        x = self.relu2(x)
        x = self.conv1(x)
        x = self.relu3(x)
        x = self.conv2(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 5, 32, 32)
