
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(7, 5)
 
    def forward(self, x1):
        v1 = self.lin(x1)
        v2 = v1 > 0.
        v3 = v3 * 0.01
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 7, 1)