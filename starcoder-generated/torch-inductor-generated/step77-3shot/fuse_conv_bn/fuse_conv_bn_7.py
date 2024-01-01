
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, 5, 1)
        self.conv2 = torch.nn.Conv2d(10, 20, 5, 1)
        self.conv3 = torch.nn.Conv2d(20, 30, 3, 1)
        self.bn1 = torch.nn.BatchNorm2d(10)
        self.bn2 = torch.nn.BatchNorm2d(20)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.conv3(output)
        return output
# Inputs to the model
x = torch.randn(1, 1, 28, 28)
