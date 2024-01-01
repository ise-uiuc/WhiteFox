
class Model(torch.nn.Module):
    def __init__(self, linear, other):
        super().__init__()
        self.linear = linear
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model
m = Model(torch.nn.Linear(10, 20), torch.rand(20))
 
# Inputs to the model
x1 = torch.randn(25, 10)
