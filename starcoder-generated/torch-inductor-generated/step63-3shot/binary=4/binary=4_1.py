
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6)
 
    def forward(self, x):
        v0 = torch.ones_like(x)
        v1 = torch.matmul(self.linear(x), v0)
        return v1, self.linear(x)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
__output__, 