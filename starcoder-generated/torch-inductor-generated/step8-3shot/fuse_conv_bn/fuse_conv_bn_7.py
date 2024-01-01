
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(5)
        w = torch.randn(2, 3).t()
        torch.manual_seed(3)
        self.bn2 = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, groups=(3))
        torch.manual_seed(4)
        self.bn3 = torch.nn.BatchNorm2d(3)
        self.conv3 = torch.nn.Conv2d(3, 2, 3, groups=(3))
        torch.manual_seed(3)
        self.bn4 = torch.nn.BatchNorm2d(2)
        self.linear = torch.nn.Linear(2, 4)
        self.linear.weight = torch.nn.Parameter(w)
    def forward(self, x):
        x = torch.nn.functional.relu(self.bn2(self.conv1(x)))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = torch.nn.functional.relu(self.bn4(self.linear(x)))
        return x
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
