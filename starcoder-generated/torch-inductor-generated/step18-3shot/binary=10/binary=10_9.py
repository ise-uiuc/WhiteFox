
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        __ = torch.nn.Linear(32, 32)
 
    def forward(self, x1): # The input tensor is x1
        v1 = __.forward(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
other = torch.randn(1, 32)
