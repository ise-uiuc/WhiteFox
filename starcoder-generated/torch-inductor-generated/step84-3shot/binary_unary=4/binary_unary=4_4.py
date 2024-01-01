
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 64)
 
    def forward(self, x, other):
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(3, 4)
other = torch.randn(4)
