
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        t1 = torch.addmm(x1, x2, x3)
        t2 = torch.cat((t1), dim)
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.tensor(10.0)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
