
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(...)
 
    def forward(self, x=?, **kwargs):
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = relu(v2)
        return v3

# Initializing the model
m = Model(...)

# Inputs to the model
x = torch.randn(1, 64, 64)
