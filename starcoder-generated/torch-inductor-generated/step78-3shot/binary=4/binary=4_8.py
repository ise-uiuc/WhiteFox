
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
 
    def forward(self, x, other):
        x = self.linear(x)
        x = x + other
        return x

# Initializing the model
m = Model()

# Inputs to the model
__x = torch.rand(1)
__other = torch.rand(1)
