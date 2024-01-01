
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 50)
 
    def forward(self, x):
        f1 = self.linear(x)
        f2 = f1 - other
        return f2

# Initializing the model
m = Model()

# Inputs to the model
f1 = torch.randn(100) # A scalar 'other'
x = torch.randn(10, 10)
