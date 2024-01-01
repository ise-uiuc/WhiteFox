
class Model(torch.nn.Module):
    def __init__(self, other=torch.empty(3, 3)):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
other = torch.randn(3, 3)
m = Model(other=other)

# Inputs to the model
x1 = torch.randn(1, 1)
