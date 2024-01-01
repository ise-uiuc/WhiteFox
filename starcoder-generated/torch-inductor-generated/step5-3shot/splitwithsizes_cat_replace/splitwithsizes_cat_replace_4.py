
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        t1, t2, t3 = torch.split(x1, 1, 2)
        x2 = torch.cat([t1, t2, t3], 2)
        return x2

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 1000, 2)
