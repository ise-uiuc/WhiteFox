
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20)
 
    def forward(self, x, other):
        t1 = self.linear(x)
        return t1 + other

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(4, 10)
other = torch.randn(4, 20)
