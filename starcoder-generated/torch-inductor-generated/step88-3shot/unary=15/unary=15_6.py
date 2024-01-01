
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 96, 3, stride=1, padding=1)
        self.maxpool1 = torch.nn.MaxPool2d(3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(96, 192, 3, stride=1, padding=1)
        self.maxpool2 = torch.nn.MaxPool2d(3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(192, 384, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(384, 256, 3, stride=1, padding=1)
        self.maxpool = torch.nn.MaxPool2d(3, stride=1, padding=1)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(256 * 3 * 3, 4096)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.fc3 = torch.nn.Linear(4096, 1)
    def forward(self, x1):
        v1 = self.maxpool1(self.conv1(x1))
        v2 = self.maxpool2(self.conv2(v1))
        v3 = self.conv3(v2)
        v4 = torch.relu(v3)
        v5 = self.conv4(v4)
        v6 = torch.relu(v5)
        v7 = self.maxpool(v6)
        v8 = self.flatten(v7)
        v9 = torch.relu(self.fc1(v8))
        v10 = torch.relu(self.fc2(v9))
        v11 = self.fc3(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 320, 320)
