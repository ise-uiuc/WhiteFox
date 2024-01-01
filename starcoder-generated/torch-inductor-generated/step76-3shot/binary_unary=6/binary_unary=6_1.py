
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
other = torch.tensor([1.0, -1.0])
m = Model(other)

# Inputs to the model
x1 = torch.randn(1, 4)
