
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x2, other):
        v1 = self.linear(x2)
        v2 = v1 + other
        v3 = relu(v2)
        return v3
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 16)
other = torch.rand(2, 32)
