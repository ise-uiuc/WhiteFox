
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(535, 379)
 
    def forward(self, x, weight=None):
        out = self.linear(x) if weight is None else self.linear(x) + weight
        return torch.nn.functional.relu(out)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 13, 19)
