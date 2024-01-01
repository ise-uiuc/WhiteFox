
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 5)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(5, 4)
        self.sigmoid1 = torch.nn.Sigmoid()
    def forward(self, x):
        z = self.fc1(x)
        x = self.relu1(z)
        x = self.fc2(x)
        y = self.sigmoid1(x)
        return y
# Inputs to the model
x = torch.randn(1, 3)
