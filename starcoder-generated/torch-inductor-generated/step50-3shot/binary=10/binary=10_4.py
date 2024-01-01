
class Model(nn.Module):
    def __init__(self, other=None):
        super().__init__()
        self.linear = torch.jit.trace(nn.Linear(5, 5), torch.zeros(5, 5))
        
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.linear.weight
        if self.other:
            v2 = v2 + self.other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
