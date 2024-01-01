
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y, z):
        t1 = torch.mm(x, y)
        t2 = t1.t() @ t1
        return t2 @ z
# Inputs to the model
x = torch.randn(3, 3)
y = torch.randn(3, 3)
z = torch.randn(3, 3)
