
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 64)
 
    def forward(self, x1, x2 = None, x3 = None):
        v1 = self.linear(x1)
        v2 = v1 + (x2 if x2 is not None else x3)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 2, 128)
x2 = torch.randn(4, 128)
x3 = torch.randn(4, 1)
