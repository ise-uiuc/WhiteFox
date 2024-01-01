
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5, bias=False)
        self.other = other
 
    def forward(self, x1):
        v1 = self.other
        v2 = self.linear(x1)
        v3 = v1 + v2
        v4 = v3.relu()
        return v4

# Initializing the model
m = Model(torch.randn(5))

# Inputs to the model
x1 = torch.randn(10, 3)
