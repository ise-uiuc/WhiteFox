
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1000, 1000)
 
    def forward(self, x, other):
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1000)
__input_other__ = torch.randn(1, 1000)
