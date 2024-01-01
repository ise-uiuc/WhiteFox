
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.Linear(14, 10)
        self.m2 = torch.nn.Linear(10, 1)
 
    def forward(self, x1):
        v1 = self.m1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 14)
# Initialize the optimizer
optimizer = torch.optim.Sgd(m.parameters(), 0.1)
# Forward pass with random weights
