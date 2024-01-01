
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
 
    def forward(self, x0, other):
        v0 = self.linear(x0)
        v2 = v0 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(4, 8)
other = torch.randn(4, 16)
