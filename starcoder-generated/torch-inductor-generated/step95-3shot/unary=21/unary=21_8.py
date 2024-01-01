
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 5, stride=3, padding=3, dilation=2, groups=1)
        self.conv2 = torch.nn.Conv2d(8, 64, 3, stride=3, padding=1, dilation=1, groups=1)
        self.fc1 = torch.nn.Linear(64, 1000)
        self.bn = torch.nn.BatchNorm2d(64)
        self.tanh = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(p=0.2)
        self.fc2 = torch.nn.Linear(1000, 1000)
        self.conv3 = torch.nn.Conv2d(64, 4, 3, stride=2, padding=1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(4, 20, 3, stride=2, padding=3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.tanh(self.bn(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.tanh(self.bn(x))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 208, 304)
