
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
 
    def forward(self, x1, other=None):
        if other is None:
            other = torch.randn(4)
 
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = v2.clamp(min=0)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
