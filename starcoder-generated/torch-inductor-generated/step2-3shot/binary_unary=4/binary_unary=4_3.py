
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
    
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model(other=torch.empty(4, 3))

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
