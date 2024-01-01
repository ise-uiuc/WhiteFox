
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v3 = v1 + self.other
        return v3

# Initializing other tensors
o1 = torch.zeros(5) 
o2 = torch.ones(5)
o3 = np.array([1.0])
o4 = np.array([2.0])
o5 = 2

# Inputs to the model
x1 = torch.randn(10, 10)
