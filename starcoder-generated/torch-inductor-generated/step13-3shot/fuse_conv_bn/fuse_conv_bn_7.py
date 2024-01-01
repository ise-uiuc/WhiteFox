
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 2)
        self.bn1 = torch.nn.BatchNorm2d(1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(1, 1, 2)
        self.relu = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(1, 1, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 8, 8)
