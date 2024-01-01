
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1, bias=True)
 
    def forward(self, x1, x2=None, x3=None):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, dtype=torch.float32)
x2 = torch.randn(1, 1, dtype=torch.float32)
