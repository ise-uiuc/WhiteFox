
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v2 = None # FILL_THIS_OUT
        v3 = None # FILL_THIS_OUT
        v4 = None # FILL_THIS_OUT
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 60, 64)
x3 = torch.randn(1, 3, 64, 64)
x4 = torch.randn(1, 3, 70, 64)
