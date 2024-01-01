
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
        self.other = other
 
    def forward(self, x):
        v1 = self.other
        v2 = self.linear(x)
        v3 = v2 - v1
        return v3.sum()

# Initializing the model
m = Model(torch.zeros([]))

# Inputs to the model
x = torch.randn(1, 4, 16)
