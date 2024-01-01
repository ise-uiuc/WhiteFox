
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1):
        s1, s2 = torch.split(x1, 8, dim=2)
        c = torch.cat([s1, s2], dim=2)
        return True if c.equal(x1) else False

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 128, 32)
