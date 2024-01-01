
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(32, 64)
 
    def forward(self, x):
        v = self.linear(x)
        v2 = v + other
        return F.relu(v2)

# Initializing the model
m = Model(other=torch.randn(1, 32))

# Inputs to the model
x = torch.randn(1, 32)
