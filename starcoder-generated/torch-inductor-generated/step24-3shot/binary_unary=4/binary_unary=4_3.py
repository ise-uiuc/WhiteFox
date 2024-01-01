
class Model(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        self.linear = torch.nn.Linear(9, 6)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + t
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model(other = torch.randn(6))

# Inputs to the model
x1 = torch.randn(1, 9)
