
_linear_pointwise = {}
n = 1
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(n):
            setattr(self, f'linear_{i}', torch.nn.Linear(8, 3))
 
    def forward(self, x):
        o = 0.0
        for _ in range(n):
            l = getattr(self, f'linear_{_}')
            o += l(x)
        return o

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
