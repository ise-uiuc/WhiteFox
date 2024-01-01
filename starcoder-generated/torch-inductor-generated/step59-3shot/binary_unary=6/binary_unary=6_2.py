
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Linear(6, 3)
 
    def forward(self, x1):
        v1 = self.mlp(x1)
        v2 = v1 - 1
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6)
