
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, other=None):
        y = self.linear(x1)
        v2 = v1 + other
        v3 = torch.relu(y)
        return y

# Initializing the model
m = Model()

# Inputs of the model
x1 = torch.randn(1, 3, 64, 64)
other = torch.randn(1, 8)
__output__= m(x1, other)
