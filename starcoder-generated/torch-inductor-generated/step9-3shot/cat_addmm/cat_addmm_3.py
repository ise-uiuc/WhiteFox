
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1):
        t1 = torch.addmm(x1, torch.randn(3, 5), torch.randn(5, 2))
        t2 = torch.cat([t1], 10)
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
