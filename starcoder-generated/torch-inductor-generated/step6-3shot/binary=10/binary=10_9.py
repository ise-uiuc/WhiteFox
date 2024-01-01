
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(9, 3)
        self.other = other
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 + self.other
        return t2

# Initializing the model
m = Model(other=torch.randn(2, 3))

# Inputs to the model
x1 = torch.randn(2, 9)
