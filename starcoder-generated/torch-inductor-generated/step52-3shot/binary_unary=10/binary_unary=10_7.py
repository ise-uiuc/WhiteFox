
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2 # Add x2 to the output of the linear transformation
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = M()

# Inputs to the model
x1 = torch.randn(1, 1) # Input tensor 1
x2 = torch.randn(1, 2) # Input tensor 2
