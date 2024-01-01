
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(48, 48)
        self.fc2 = torch.nn.Linear(48, 48)
        self.fc3 = torch.nn.Linear(48, 37)
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = self.fc2(v1)
        v3 = self.fc3(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 37, 48)
