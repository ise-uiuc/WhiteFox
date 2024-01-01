
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return v1 + other

# Initializing the model
m = Model()

# Inputs to the model, the shape of "other" can be [1, 8]
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 100, 10)
__output1__ = m(x1)
__output2__ = m(x2)

