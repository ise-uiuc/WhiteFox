
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 4)
 
    def forward(self, x1):
        o1 = x1
        o2 = self.lin(o1)
        o3 = o2 * 0.5
        o4 = o2 * 0.7071067811865476
        o5 = torch.erf(o4)
        o6 = o5 + 1
        o7 = o3 * o6
        return o7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8)
