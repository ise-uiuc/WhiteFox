
class Model(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
 
    def forward(self, x):
        return self.linear(x) + x

# Initializing the model
m = Model(3, 8)

# Inputs to the model
x1 = torch.randn(1, 3)
