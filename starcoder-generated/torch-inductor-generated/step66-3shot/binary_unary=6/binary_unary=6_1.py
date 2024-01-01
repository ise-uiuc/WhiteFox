
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - other
        v3 = v2 + 2.0 * const
        v4 = x1 * (-1 * v3)
        v6 = v4 + 1
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20)
