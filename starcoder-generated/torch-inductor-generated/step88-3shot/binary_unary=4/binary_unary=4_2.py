
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
 
    def forward(self, x1, other):
        x2 = self.linear(x1)
        x3 = x2 + other
        return torch.relu(x3)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 5, 5)
other = torch.randn(1, 2, 5, 5)
