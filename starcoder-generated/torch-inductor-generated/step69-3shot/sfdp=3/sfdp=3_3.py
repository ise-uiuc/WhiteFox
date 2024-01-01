
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 2)
        self.fc2 = torch.nn.Linear(3, 2)
        self.fc3 = torch.nn.Linear(2, 1)
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = self.fc2(v1)
        v3 = v2.mul(10)
        v4 = self.fc3(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)

# Inputs to the model
x1 = torch.randn(1, 4)
x2 = torch.randn(1, 3)
x3 = torch.randn(1, 2)
