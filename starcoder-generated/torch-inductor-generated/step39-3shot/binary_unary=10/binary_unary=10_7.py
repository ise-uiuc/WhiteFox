
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(42, 57)

    def forward(self, x2):
        v1 = self.lin(x2)
        v2 = v1 + torch.randn(57)
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(23, 42)
