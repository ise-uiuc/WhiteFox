
class Model(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 - x2
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model(8, 4)

# Inputs to the model
x1 = torch.randn(3, 8, 64, 64)
x2 = torch.randn(3, 4, 64, 64)
