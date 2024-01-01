
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
        self._other_tensor = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self._other_tensor
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
other = torch.randn(1, 16)
m = Model(other)

# Inputs to the model
x1 = torch.randn(1, 8)
