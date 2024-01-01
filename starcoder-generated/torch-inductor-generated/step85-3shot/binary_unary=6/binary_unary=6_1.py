
class Model(torch.nn.Module):
    def __init__(self, in_dim, out_dim, other):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model(in_dim=10, out_dim=3, other=5)

# Inputs to the model
x1 = torch.randn(1, 10)
