
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 2)
        self.fc2 = torch.nn.Linear(2, 2)
        self.conv1 = torch.nn.Linear(2, 2)
        self.conv2 = torch.nn.Linear(2, 2)
        self.fc3 = torch.nn.Linear(2, 2)
        self.fc4 = torch.nn.Conv2d(3, 7, 1, stride=1)
    def forward(self, x1, x2):
        v1 = self.fc1(torch.add(x1, x2) * x1)
        v2 = self.fc2(torch.cat((x1, x2), dim=1))
        v3 = torch.flatten(v2, 1)
        v4 = self.conv1(torch.abs(torch.add(x1,x2) * x2))
        v5 = torch.cat((self.conv2(x2), v2), dim=1)
        v6 = v3 + torch.flatten(v5, 1)
        v7 = self.fc3(torch.reshape(torch.add(v1, x2), (2, 2)))
        v8 = v6 + self.fc4(v5 + v4)
        v9 = self.fc4(v7 + v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 2)
