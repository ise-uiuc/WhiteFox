
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 1, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return torch.clamp_min(v1, min_value=-1.0)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
__output = m(x1)
