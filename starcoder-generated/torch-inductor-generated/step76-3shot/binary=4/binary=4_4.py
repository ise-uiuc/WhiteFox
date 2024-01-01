
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(200, 200)
 
    def forward(self, x1, other1):
        v1 = self.linear(x1)
        v2 = v1 + other1
        return v2

# Initializing the model
w = torch.randn(2, 3)
b = torch.randn(2)
m = Model()

# Inputs to the model
x1 = torch.randn(10, 20)
other1 = torch.randn(10, 2)
__outputs__ = m(x1, other1)
