
class Model(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.activations = torch.nn.GELU()
 
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = v1 + x
        v3 = self.activations(v2)
        return v3

# Initializing the model
m = Model(hidden_dim=16)

# Inputs to the model
x = torch.randn(2, 16)
