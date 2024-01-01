
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 16)
 
    def forward(self, x1, other):
        t1 = self.linear(x1)
        t2 = t1 + other
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
x2 = torch.randn(2, 3, 64, 64)
x2 = torch.zeros(2, 3, 64, 64)
print(x1)
