
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(5, 6)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model(torch.randn(5, 6))

# Inputs to the model
x1 = torch.randn(2, 5)
