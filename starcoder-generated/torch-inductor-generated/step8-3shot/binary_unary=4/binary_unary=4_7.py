
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(16, 2)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model with a tensor
other_tensor = torch.randn(2)
m = Model(other=other_tensor)

# Inputs to the model
x1 = torch.randn(2, 16)
