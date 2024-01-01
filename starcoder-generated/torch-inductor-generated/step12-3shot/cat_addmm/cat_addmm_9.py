
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(96, 72)
        self.fc1 = torch.nn.Linear(72, 12)
        self.fc2 = torch.nn.Linear(12, 3)
 
    def forward(self, x1):
        v1 = torch.cat([self.fc(x1), self.fc1(x1), self.fc2(x1)], dim=0)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 96)
