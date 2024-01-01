
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool2d = torch.nn.MaxPool2d(2, stride=1)
        self.conv1 = torch.nn.Conv2d(3, 32, 8)
        self.conv2 = torch.nn.Conv2d(32, 32, 4, padding=2)
        self.conv3 = torch.nn.Conv2d(32, 8, (3, 3), 1, 0)
    def forward(self, x):
        x = self.maxpool2d(self.relu(self.conv1(x)))
        x = self.maxpool2d(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
