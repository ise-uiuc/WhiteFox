
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 10)
    def forward(self, x1):
        v1 = torch.reshape(x1, (-1, 28*28))
        v2 = self.fc1(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.fc2(v3)
        v5 = torch.relu(v4)
        v6 = self.fc3(v5)
        v7 = torch.sigmoid(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
