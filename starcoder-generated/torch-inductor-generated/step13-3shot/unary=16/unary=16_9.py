
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(10, 10)
        self.lin3 = torch.nn.Linear(10, 10)
 
    def forward(self, x1):
        v1 = self.lin1(x1)
        v2 = torch.relu(v1)
        v3 = self.lin3(v2)
        v4 = torch.tanh(v3)
        v5 = torch.relu(v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
