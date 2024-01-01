
class Model(torch.nn.Module):
    def __init__(self, other1):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)
        self.other1 = other1
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 + self.other1
        return torch.nn.functional.relu(t2)

# Initializing the model
other1 = list(torch.Tensor([range(100)]))
m = Model(torch.nn.Parameter(other1[0]))

# Inputs to the model
x1 = torch.randn(1, 100)
