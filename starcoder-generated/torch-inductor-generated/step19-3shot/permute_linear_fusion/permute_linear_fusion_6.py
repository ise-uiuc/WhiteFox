
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.mp1 = torch.nn.MaxPool2d(2, 2)
        self.mp2 = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(800, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        x = self.mp1(x)
        x = F.relu(self.conv1(x))
        x = self.mp2(x)
        x = F.relu(self.conv2(x))
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
# Inputs to the model
x = torch.randn(1, 1, 28, 28)
