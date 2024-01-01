
class Model(torch.nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.linear = torch.nn.Linear(size, 1)
 
    def forward(self, x0):
        v0 = x0 * x0
        v1 = self.linear(v0)
        other = 1
        if v1 >= 0:
            other = -1
        v2 = v1 + other
        return v2

# Initializing the model
size = 256 * 256
m = Model(size)

# Inputs to the model
x0 = torch.randn(1, size)
