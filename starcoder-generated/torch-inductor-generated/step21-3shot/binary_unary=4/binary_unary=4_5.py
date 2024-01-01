
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 6)
 
    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        if other is None:
            other = torch.ones_like(v1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3


# Initializing the model
m = Model()

# Inputs and keyword arguments to the model
x1 = torch.randn(3, 4)
