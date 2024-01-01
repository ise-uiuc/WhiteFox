
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=8, out_features=16)
 
    def forward(self, x6):
        v1 = torch.flatten(x6, 1)
        v2 = self.lin(v1)
        v3 = v2 - 3
        v4 = torch.nn.functional.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x6 = torch.randn(1, 8)
