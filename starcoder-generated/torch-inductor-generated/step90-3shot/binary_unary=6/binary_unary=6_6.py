
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.other = torch.nn.Parameter(torch.randn(8))

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Intializing the other Parameter
m.other = torch.nn.Parameter(torch.randn(8))

# Inputs to the model
x1 = torch.randn(1, 8)
