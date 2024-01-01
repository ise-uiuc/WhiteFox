
class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.relu(v1)
        return v2
 
# Initializing the model
dims = (10, 10, 10)
m = Model(*dims)

# Inputs to the model
x1 = torch.randn(*dims)
