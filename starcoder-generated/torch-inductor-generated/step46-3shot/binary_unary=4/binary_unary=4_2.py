
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = relu(v2)
        return v3

# Initializing the model
m = Model(torch.randn(1, 32))

# Inputs to the model
x = torch.randn(1, 32)
