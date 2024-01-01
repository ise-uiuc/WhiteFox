
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
other = torch.randn(16, 16)
x1 = torch.randn(16, 16)
