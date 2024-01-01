
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 4, 1, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 4, 3, stride=1)
        self.conv4 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.fc1 = torch.nn.Linear(1024, 64)
        self.fc2 = torch.nn.Linear(64, 64)
    def forward(self, x1, x2):
        v0 = self.conv3(x1)
        v1 = self.conv4(x2)
        v2 = self.fc1(v0)
        v3 = torch.sum(v1)
        v4 = self.fc2(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
x2 = torch.randn(1, 3, 28, 28)
