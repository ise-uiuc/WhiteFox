
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 32)
 
    def forward(self, x1, x2=None):
        if not x2:
            return self.linear(x2)
        o = self.linear(x1)
        o += x2
        return torch.nn.functional.relu(o)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
x2 = torch.randn(1, 32)

