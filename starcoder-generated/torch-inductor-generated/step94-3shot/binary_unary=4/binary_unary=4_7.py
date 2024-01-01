
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(12, 10)
 
    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3

# Initializing the model
other = torch.randn(10)
m = Model(other=other)

# Inputs to the model
x1 = torch.randn(12)
