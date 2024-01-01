
class _MM(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, a, b):
        return torch.mm(a, b)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm1 = _MM()
        self.mm2 = _MM()
 
    def forward(self, x, y):
        v1 = self.mm1.forward(x, y)
        v2 = self.mm2.forward(x, y)
        v3 = v1 + v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 2, 4)
y = torch.randn(1, 8, 4, 1)
