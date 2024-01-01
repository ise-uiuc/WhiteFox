
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.other = torch.nn.Parameter(other)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2, v1
 
# Initializing the model
tensor_other = torch.randn(2, 4)
m = Model(tensor_other)

# Inputs to the model
x1 = torch.randn(1, 4)
__output__, __tmp__ = m(x1)

