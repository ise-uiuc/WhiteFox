
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10, bias=False)
 
    def forward(self, x, other=None):
        v1 = self.linear(x)
        v2 = v1
        if other is not None:
            v2 = v2 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 5)
other = None # The value can be changed as desired
