
class Model(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear layer = torch.nn.Linear(in_dim, out_dim)
 
    def forward(self, x1, x2):
        x = self.linear(x1)
        x = x + x2
        x = torch.tanh(x)
        x = self.linear(x)
        x = x + x2
        return x2

# Initializing the model
m = Model(in_dim, out_dim)

# Inputs to the model
x1 = torch.randn(1, in_dim)
x2 = torch.randn(1, in_dim)
