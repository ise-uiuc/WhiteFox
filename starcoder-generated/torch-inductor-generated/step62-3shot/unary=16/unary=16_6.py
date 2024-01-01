
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(300, 100)
        self.fc2 = torch.nn.Linear(100, 40)
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = self.fc2(F.relu(v1))
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 300)
