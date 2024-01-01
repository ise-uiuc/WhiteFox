
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 1)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
m = Model()
m.weight = torch.nn.Parameter(torch.ones([1, 8]))
m.bias = torch.nn.Parameter(torch.ones([1]))

# Inputs to the model
x1 = torch.randn(1, 8)
x2 = torch.randn(1, 1)
