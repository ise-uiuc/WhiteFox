
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.bn2 = torch.nn.BatchNorm2d(50)
        self.max_pool2d = torch.nn.MaxPool2d(2, 2)
        self.view = lambda x: x.view(x.shape[0], -1)
        self.drop_out = torch.nn.Dropout()
        self.linear1 = torch.nn.Linear(4 * 4 * 50, 500)
        self.bn3 = torch.nn.BatchNorm1d(500)
        self.linear2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.max_pool2d(self.bn2(self.conv2(x)))
        x = self.view(x)
        x = self.drop_out(x)
        x = self.bn3(self.linear1(x))
        x = self.relu(x)
        x = self.linear2(x)
        return x
# Inputs to the model
x = torch.randn(2, 1, 28, 28)
