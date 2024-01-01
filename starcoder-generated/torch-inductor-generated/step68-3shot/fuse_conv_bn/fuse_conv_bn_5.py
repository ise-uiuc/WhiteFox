
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        torch.manual_seed(0)
        self.bn1 = torch.nn.BatchNorm2d(20)
        torch.manual_seed(0)
        self.conv2 = torch.nn.Conv2d(20, 64, 5, 1)
        torch.manual_seed(0)
        self.bn2 = torch.nn.BatchNorm2d(64)
        torch.manual_seed(0)
        self.fc1 = torch.nn.Linear(9216, 128)
        torch.manual_seed(0)
        self.bn3 = torch.nn.BatchNorm1d(128)
        torch.manual_seed(0)
        self.fc2 = torch.nn.Linear(128, 10)
    def forward(self, x):
        y = self.conv1(x)
        o = self.bn1(y)
        z = self.conv2(o)
        o = self.bn2(z)
        o = torch.flatten(o, 1)
        o = self.bn3(o)
        o = self.fc1(o)
        o = self.bn3(o)
        o = self.fc2(o)
        return o
# Inputs to the model
x = torch.randn(28, 28, 1)
