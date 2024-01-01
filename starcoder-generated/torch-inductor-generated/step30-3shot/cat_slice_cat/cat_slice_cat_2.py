
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        z1 = torch.flip(x1,(0,))
        return x1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 256, 384)
