
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, v0):
        t1 = torch.mm(x1, x2)
        t2 = v0 + t1
        return t2
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3)
v0 = torch.randn(3, 3)
