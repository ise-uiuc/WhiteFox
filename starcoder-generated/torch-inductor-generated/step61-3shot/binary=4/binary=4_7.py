
t = torch.rand(1, 5) # Create a tensor representing the values of a certain constant distribution
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + t
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
