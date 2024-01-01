
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 1)
 
    def forward(self, x1):
        v1 = self.lin(x1)
        v2 = F.relu(v1)
        v3 = torch.sigmoid(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
