
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, other):
        return (self.linear(x1) + other)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(8)
