
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 20)
        self.fc2 = torch.nn.Linear(20, 2)
        self.fc3 = torch.nn.Linear(2, 1)
    def forward(self, x1):
        x = F.relu(self.fc1(x1))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
# Inputs to the model
x1 = torch.randn(1, 4)
