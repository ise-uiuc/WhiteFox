
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)
 
    def forward(self, x1, x2=None):
        v1 = self.linear(x1)
        if x2 is not None:
            v2 = v1 + x2
        else:
            v2 = v1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 2)
