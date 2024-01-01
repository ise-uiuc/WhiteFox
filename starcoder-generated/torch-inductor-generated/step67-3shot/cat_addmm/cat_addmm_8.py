
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 6)
        self.fc2 = nn.Linear(6, 4)
        self.fc3 = nn.Linear(4, 6)
        self.fc4 = nn.Linear(6, 1)
    def forward(self, x):
        z1 = F.relu(self.fc1(x))
        z2 = F.relu(self.fc2(z1))
        z3 = F.relu(self.fc3(z2))
        z4 = F.relu(self.fc4(z3))
        return z4
# Inputs to the model
x = torch.randn(1, 1)
