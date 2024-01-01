
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(64*64, 64)
 
    def forward(self, x1):
        v1 = self.lin(x1)
        v2 = v1 - 53
        v3 = v2.relu()
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64*64)
