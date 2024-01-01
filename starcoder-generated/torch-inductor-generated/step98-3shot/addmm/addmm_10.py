
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4):
        t1 = torch.mm(x1, x2)
        t2 = x3 + t1
        t3 = t2 + x4
        return t3
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3, requires_grad=True)
x4 = torch.randn(3, 3, requires_grad=True)
