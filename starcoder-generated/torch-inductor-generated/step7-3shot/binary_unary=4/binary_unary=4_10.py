
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 2)
 
    def forward(self, arg=None, x1=None, arg1=None, x2=None):
        v1 = self.linear(x1)
        v2 = v1 + arg
        v3 = F.relu(v2)
        return v3, v1 + x2, v1 + 1

# Initializing the model
m = Model()

# Inputs to the model
arg = torch.randn(1, 2)
x1 = torch.randn(3, 5)
arg1 = torch.randn(1, 2)
x2 = torch.randn(1, 2)
__output__, __other_output__, __final_output__ = m(arg=arg, x1=x1, arg1=arg1, x2=x2)

