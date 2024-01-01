
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 3)
 
    def forward(self, x1, x2, other):
        v1 = self.linear(x1)
        v2 = v1 + torch.nn.functional.relu(x2)
        return v2 + other

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.ones(1, 10)
