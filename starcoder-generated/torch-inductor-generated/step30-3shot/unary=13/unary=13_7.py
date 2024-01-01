
class SigmoidGated(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = torch.nn.Linear(self.dim, self.dim)
 
    def forward(self, x):
        t1 = self.linear(x)
        t2 = torch.sigmoid(t1)
        t3 = t1 * t2
        return t3

# Initializing the model
dim = 16

m = SigmoidGated(dim)

# Input to the model
x = torch.randn(32, dim, 4, 4)
