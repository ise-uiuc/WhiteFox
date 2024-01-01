
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 6,  bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.pool1 = torch.nn.MaxPool2d(3)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pool2 = torch.nn.AvgPool2d(2)
        self.fc = torch.nn.Linear(9216, 10)
    def forward(self, x):
        v1 = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        v2 = self.pool1(v1)
        v3 = torch.nn.functional.relu(self.bn2(self.conv2(v2)))
        v4 = self.pool2(v3)
        v4 = v4.view(v4.size(0), -1)
        v5 = self.fc(v4)
        return v5

# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
