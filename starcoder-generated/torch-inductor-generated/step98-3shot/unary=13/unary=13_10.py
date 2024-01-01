
class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = torch.nn.Linear(dim, 4*dim)
 
    def forward(self, x1):
        v1 = self.proj(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model(256)

# Inputs to the model
x1 = torch.randn(1, 256)
