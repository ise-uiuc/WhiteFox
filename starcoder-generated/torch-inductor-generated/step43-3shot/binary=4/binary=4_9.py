
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, other = None):       
        v0 = torch.ones((1, 1), device=x1.device, dtype=x1.dtype)
        v1 = v0 * x1
        if other is not None:
            v2 = self.linear(v1) + other
        else:
            v2 = self.linear(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
m(x1)

