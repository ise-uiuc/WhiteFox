
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.other = torch.empty((8,))
        torch.nn.init.uniform_(self.other)
 
    def forward(self, x1, x2=None, z=2):
        if x2 is None:
            x2 = self.other
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = F.relu(v2, inplace=False)
        v4 = v3 + z
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 8)  # A keyword argument
