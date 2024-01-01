
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d((3, 2), stride=1)
        self.conv1 = torch.nn.Conv2d(3, 64, (1, 1), stride=(1, 1))
        self.conv2 = torch.nn.Conv2d(64, 128, (3, 3), stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 128, (1, 1), stride=(1, 1))
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
    def forward(self, x):
        x = self.relu(self.conv1(self.avgpool(x)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        return x
# Inputs to the model
x = torch.randn(3, 1, 56, 56)
