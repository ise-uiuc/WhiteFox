
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.other = torch.randn(8, 3)
 
    def forward(self, x2, __other__=None):
        v1 = self.linear(x2)
        v2 = v1 + __other__
        v3 = v2.relu()
        return v3
 
# Initializing the model
m = Model()

# Initializing the keyword argument
other0 = torch.randn(8, 3)
other = [other0]

# Inputs to the model
x2 = torch.randn(1, 3)
