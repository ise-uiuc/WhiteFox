
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
 
    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        v2 = self.linear(other)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
__input2__ = (torch.randn(1, 1) if bool(random.getrandbits(1)) else torch.ones(1, 1))
