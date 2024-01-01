
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1):
        v1 = torch.randn(x1)
        v2 = v1.t()
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 10, 20)
