
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(1, 1)
        self.other = torch.nn.Parameter(torch.ones(1, 1))
 
    def forward(self, x1, other=None):
        if other is None: other_local = self.other
        else: other_local = other
        v1 = self.l1(x1)
        v2 = v1 + other_local
        return torch.nn.functional.relu(v2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
