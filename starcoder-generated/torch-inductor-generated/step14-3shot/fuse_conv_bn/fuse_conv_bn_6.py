
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 3)
        self.conv2 = torch.nn.Conv2d(3, 1, 3)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU()
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        y = x + self.bn(x)
        return y
# Inputs to the model
x1 = torch.randn(1, 3, 6, 6)
x2 = torch.randn(1, 3, 2, 2)
