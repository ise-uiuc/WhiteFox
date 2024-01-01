
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 2, 1)
        self.conv2 = torch.nn.Conv2d(2, 2, 3)
        self.bn = torch.nn.BatchNorm2d(2)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 2, 4, 4)
