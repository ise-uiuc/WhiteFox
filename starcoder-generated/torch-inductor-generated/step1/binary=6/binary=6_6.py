
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
 
    def forward(self, x, other=None):
        y = x
        if other is not None:
            v1 = torch.sub(y, other)
        else:
            v2 = torch.sub(y, y)
            v3 = torch.sub(y, y)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
