
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
        self.negative_slope = torch.nn.Parameter(torch.Tensor([0.01])) # Only the last element is nonzero. The other elements are all zero.
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 > 0
        t3 = t1 * self.negative_slope
        t4 = torch.where(t2, t1, t3)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
