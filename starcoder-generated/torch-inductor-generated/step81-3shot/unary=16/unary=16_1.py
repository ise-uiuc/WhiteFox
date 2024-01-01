
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.relu(v1)
        v3 = torch.erf(v1)
        v4 = v2 * v1
        v5 = v4 + 1
        return v5

# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(3, 2)
