
class LinearModel(torch.nn.Module):
    def __init__(self, size1, x2):
        self.linear = torch.nn.Linear(len(size1))
        self.x2 = x2
        __out_size__ = len(size1)
    def forward(self, x1):
        __temp__ = len(x1)
        return self.linear(x1 + self.x2), __out_size__

# Initializing the model
m = LinearModel(size1=[2,2,2], x2=[1])

# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
__output__, out_size = m(x1)

