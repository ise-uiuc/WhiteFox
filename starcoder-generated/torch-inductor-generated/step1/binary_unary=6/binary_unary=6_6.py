
class Model(torch.nn.Module):
    def __init__(self, other=0):
        super().__init__()
        self.w = torch.nn.Parameter(torch.rand(3, 3))
        self.other1 = other
 
    def forward(self, x):
        v1 = torch.nn.functional.linear(x, self.w)
        v2 = v1 - self.other1
        return torch.nn.functional.relu(v2)

# Initializing the model
m = Model(0.1)

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
