
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, x1):
        o = self.linear(x1)
        r = o - 2
        e = torch.relu(r)
        return e

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
