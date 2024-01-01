
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y, z):
        z1= torch.mm(x, y)
        z2= torch.mm(x, z)
        z3= torch.mm(y, z)
        t1 = z1 + z2
        t2 = t1 + z3
        return t2
# Inputs to the model
x = torch.randn(10, 10)
y = torch.randn(10, 10)
z = torch.randn(10, 10)
