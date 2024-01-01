
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.l = torch.nn.Linear(1, 1)
 
    def forward(self, x1):
        v1 = self.l(x1)
        v2 = v1 - other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model(torch.tensor(2).float())

# Inputs to the model
x1 = torch.randn(1, 1)
