
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(28, 10)
 
    def forward(self, x1):
        v1, = self.lin.get_weight() 
        v2 = torch.sum(v1)
        v3 = v2 * other
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 28)
