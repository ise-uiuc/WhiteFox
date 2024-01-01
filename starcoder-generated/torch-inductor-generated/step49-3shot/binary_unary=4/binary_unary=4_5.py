
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model(other = torch.randn(1, 8))

# Inputs to the model
x2 = torch.randn(1, 8)
