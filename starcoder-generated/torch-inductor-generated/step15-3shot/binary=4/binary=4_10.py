
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
     def forward(self, x1, other=None):
        if other is not None:
            v1 = self.linear(x1)
            return v1 + other
        else:
            v1 = self.linear(x1)
            return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 10)
x2 = torch.randn(2, 10)

