
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6)
 
    def forward(self, x1):
        self.linear(x1)
        v1 = self.linear(x1)
        v2 = v1 + 3
        self.linear(v2)
        v3 = torch.nn.functional.hardsigmoid(v2)
        v4 = 6 * v3
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
