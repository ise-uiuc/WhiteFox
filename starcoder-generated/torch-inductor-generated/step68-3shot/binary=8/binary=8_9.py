
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.fc1 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.fc2 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.fc3 = torch.nn.Conv2d(3, 4, 1, stride=1)
    def forward(self, x1, x2):
        v5 = self.fc2(x2)
        v7 = self.fc3(x1)
        v3 = self.conv1(x1)
        v6 = v5 + v7
        v9 = self.fc1(x2)
        v11 = self.fc1(x2)
        v4 = v6 + v3
        v8 = self.fc2(x1)
        v10 = v9 + v8
        v12 = self.fc3(x2)
        v14 = self.fc3(x2)
        v13 = v11 + v10
        return (v4, v13)
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
x2 = torch.randn(1, 3, 16, 16)
