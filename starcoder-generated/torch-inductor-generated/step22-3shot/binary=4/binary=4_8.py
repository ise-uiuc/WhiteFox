
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 2)
 
    def forward(self, x1, other):
        return self.linear(x1) + other

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
other = torch.randn(2)
