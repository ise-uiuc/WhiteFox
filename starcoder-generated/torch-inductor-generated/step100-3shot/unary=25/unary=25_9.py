
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 3, bias=False)
        self.lrelu = torch.nn.LeakyReLU()
 
    def _make_param(slef, dims):
        return torch.nn.Parameter(torch.zeros(dims))
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 > 0
        v3 = self._make_param((-1,))
        v4 = torch.where(v2, v1, v3)
        return self.lrelu(v4)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 32)
