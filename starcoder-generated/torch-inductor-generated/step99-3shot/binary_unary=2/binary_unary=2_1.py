
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.conv4 = torch.nn.Conv2d(32, 16, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = self.pool1(v2)
        v4 = self.conv2(v3)
        v5 = F.relu(v4)
        v6 = self.conv3(v5)
        v7 = F.relu(v6)
        v8 = self.pool2(v7)
        v9 = self.conv4(v8)
        v10 = F.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
