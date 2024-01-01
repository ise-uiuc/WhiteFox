
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.l = torch.nn.Linear(4, 4)
        self.other = other
    
    def forward(self, x1):
        v1 = self.l(x1)
        v2 = v1 + self.other
        v3 = F.relu(v2)
        return v3

# Initializing the model
other = torch.ones(4, 4)
m = Model(other=other)

# Inputs to the model
x1 = torch.randn(2, 4)
