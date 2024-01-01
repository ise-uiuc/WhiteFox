
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 - other
        v3 = relu(v2)
        return v3

# Initializing the model, other as a scalar value
other = 10
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
