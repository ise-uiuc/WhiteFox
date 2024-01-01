
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 8)
 
    def forward(self, x2):
        t1 = self.lin(x2)
        t2 = t1 + x2
        v = self.relu(t2)
        return v

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3)
other = torch.empty(1, 3).uniform_(-10, 10) # A random tensor with the same shape as x2
