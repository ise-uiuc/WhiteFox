
class Model(torch.nn.Module):
    def __init__(self, _min, _max):
        super(Model).__init__()
        self.linear = torch.nn.Linear(32, 32, bias=False)
        self.min = _min
        self.max = _max
 
    def forward(self, x):
        v = linear(x)
        v2 = torch.min_(v, self.min)
        v3 = torch.min_(v2, self.max)
        return v3

# Initializing the model
m = Model(0.1, 0.8)

# Inputs to the model
x = torch.randn(1, 32)
