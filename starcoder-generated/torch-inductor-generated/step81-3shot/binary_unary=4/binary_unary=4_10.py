
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.input = torch.nn.Linear(3, 32)
 
    def forward(self, x, other=None):
        v1 = self.input(x)
        if (other is not None):
            v1 = v1 + other
        v2 = torch.nn.functional.relu(v1)
        return v2

# Initializing the model
m = Model(x1)

# Inputs to the model
x1 = torch.randn(3, 3)
