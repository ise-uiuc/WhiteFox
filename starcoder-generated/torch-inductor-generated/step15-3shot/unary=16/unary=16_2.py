
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 100)
        self.relu = torch.nn.ReLU()

    def forward(self, x2):
        v1 = self.lin(x2)
        v2 = self.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 100)
