
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.addmm(x1, x2, x1)
        v2 = v1.flatten(0)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.rand(64, 64)
x2 = torch.rand(64, 64)
