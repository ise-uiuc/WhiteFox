
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        if other is None:
            return v1
        return v1 + other

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
other = torch.randn(1, 8) if not isinstance(m, torch.nn.Sequential) else None
