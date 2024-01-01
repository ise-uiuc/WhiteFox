
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4, 1, 2, 1, True)
 
    def forward(self, x, other):
        v1 = self.linear(x)
        v2 = v1 + other
        return v2, v2 * other

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
other = torch.randn(1, 4)

# Outputs of the model
