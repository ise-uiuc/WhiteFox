
class Model(torch.nn.Module):
    other = None
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model
m = Model()

# Initializing the tensor "other" to be added
m.other = torch.randn(8, 8)

# Inputs to the model
x1 = torch.randn(1, 16)
