
class Model(torch.nn.Module):
    def __init__(self, other=None):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, x, other):
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = F.relu(t2)
        return v2

# Initializing the model
m = Model(other=torch.randn(3))

# Inputs to the model
x1 = torch.randn(1, 3)
