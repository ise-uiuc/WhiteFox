
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16, bias=True)
 
    def forward(self, x1, x2=None):
        v1 = self.linear(x1)
        if x2 is not None:
            v1 = v1 + x2
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
x2 = torch.randn(1, 16)
