
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)
 
    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        if other is not None:
            v2 = v1 + other
            v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(7, 5)
x2 = torch.randn(7, 3)
