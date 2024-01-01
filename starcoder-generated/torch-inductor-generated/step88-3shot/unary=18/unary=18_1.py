
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def flatten(self, x1, x2):
        return x1.view(-1, x1.shape[1]), x2.view(-1, x2.shape[1])

    def forward(self, x1, x2):
        x1, x2 = self.flatten(x1, x2)
        v1 = self.conv1(x1)
        v1 = F.relu(v1)
        v1 = self.pool(v1)
        v1 = self.conv2(v1)
        v1 = F.relu(v1)
        v1 = self.pool(v1)
        v1 = v1.view(-1, 16 * 53 * 53)
        v1 = self.fc1(v1)
        v1 = F.relu(v1)
        v1 = self.fc2(v1)
        v1 = F.relu(v1)
        v1 = self.fc3(v1)
        return v1
model = Net()
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
