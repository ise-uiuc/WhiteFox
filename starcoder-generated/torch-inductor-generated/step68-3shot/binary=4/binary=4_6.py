
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No additional parameter is required
 
    def forward(self, x1, x2):
        v1 = torch.nn.functional.linear(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 10)
