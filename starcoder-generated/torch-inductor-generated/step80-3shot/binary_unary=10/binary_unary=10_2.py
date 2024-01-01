
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.weight()
        v3 = torch.relu(v2)
        return v3
 
    def weight(self):
        return torch.tensor(1.), torch.tensor(1.), torch.tensor(1.)

# Initializing the model
m = Model()

# Input tensor
x1 = torch.randn(1, 10)
