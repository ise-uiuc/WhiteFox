
class Model(torch.nn.Module):
    def __init__(self):
        super(Sequential, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, 1, 1)
        self.norm = torch.nn.BatchNorm2d(16)
        relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(16, 16)
        self.fc2 = torch.nn.Linear(16, 10)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.norm(v1)
        v3 = relu(v2)
        v4 = torch.flatten(v3, 1)
        v5 = self.fc1(v4)
        v6 = relu(v5)
        v7 = self.fc2(v6)
        v8 = relu(v7)
        return v8
# Inputs to the model
x = torch.randn(8, 1, 28, 28)
