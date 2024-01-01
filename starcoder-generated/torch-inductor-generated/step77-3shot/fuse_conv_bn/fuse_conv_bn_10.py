
class Model3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 64, 5, 1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.fc = torch.nn.Linear(256, 10)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        output = self.fc(self.relu(self.bn1(self.conv2(self.relu(self.bn(self.conv1(x)))))).flatten())
        return output
# Inputs to the model
x = torch.randn(1, 1, 28, 28)
