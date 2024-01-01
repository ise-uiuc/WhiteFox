
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3

# Initializing the model with a keyword argument passed to `forward`
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
other = torch.randn(1, 16)
