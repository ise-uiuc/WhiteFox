
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(64, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = relu(v2)
        return v3

# Initializing the model
m = Model(other=torch.randn(1, 64))

# Inputs to the model
x1 = torch.randn(1, 64)
