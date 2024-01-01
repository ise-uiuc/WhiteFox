
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, 16)
        self.fc4 = torch.nn.Linear(16, 1)
 
    def forward(self, x1):
        v = self.fc1(x1).relu()
        v = self.fc2(v).relu()
        v = self.fc3(v).relu()
        v = self.fc4(v).relu()
        return v

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
x2 = torch.randn(1, 16)
