
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
 
    def forward(self, x):
        x = x.view(-1, 16 * 5 * 5)
        v1 = self.fc1(x)
        v2 = F.relu(v1)
        v3 = self.fc2(v2)
        v4 = F.relu(v3)
        v5 = self.fc3(v4)
        v6 = F.log_softmax(v5, dim=1)
        return v6

# Initializing the model
m = Net()

# Inputs to the model
x = torch.randn(1, 1, 28, 28)
