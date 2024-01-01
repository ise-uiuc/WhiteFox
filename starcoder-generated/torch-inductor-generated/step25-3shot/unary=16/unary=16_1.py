
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(10, 10)
 
    def forward(self, x1):
        a1 = self.lin(x1)
        a2 = torch.relu(a1)
        return a2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 10)
