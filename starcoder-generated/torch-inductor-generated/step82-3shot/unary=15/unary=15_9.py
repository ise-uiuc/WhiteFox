
class Model(torch.nn.Module):
    def __init__(self):
        super(torch.nn.Module, self).__init__()
        self.conv1 = torch.nn.Linear(500, 1000)
        self.conv2 = torch.nn.Conv2d(1000, 4096, 1)
        self.bn1 = torch.nn.BatchNorm2d(4096)
        self.fc1 = torch.nn.Linear(4096, 1000)
        self.fc2 = torch.nn.Linear(1000, 300)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.relu(v2)
        v4 = self.bn1(v3)
        v5 = torch.reshape(v4, [-1, 1000])
        v6 = torch.relu(v5)
        v7 = self.fc1(v6)
        v8 = torch.relu(v7)
        v9 = self.fc2(v8)
        v10 = torch.softmax(v9, 1)
        return v10
# Inputs to the model
x1 = torch.randn(1, 500)
