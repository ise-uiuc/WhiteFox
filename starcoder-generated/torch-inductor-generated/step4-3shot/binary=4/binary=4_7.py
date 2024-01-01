
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.other = other
 
    def forward(self, x1):
        intermediate = self.linear(x1)
        return intermediate + self.other

# Initializing the model
m = Model(torch.randn(10))

# Inputs to the model
x1 = torch.randn(1, 10)
