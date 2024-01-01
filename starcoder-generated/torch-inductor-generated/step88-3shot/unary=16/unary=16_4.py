 (the model may be different from the previous one)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        h1 = int(x1.size()[2] / 2)
        
        v1 = torch.nn.functional.interpolate(x1, size=(h1, h1))
        v2 = torch.randn(1, 3, 64, 64)
        
        return v2 * v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
