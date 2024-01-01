
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        t1 = x3 + x2
        t2 = x3 + t1
        t3 = torch.mm(x1, x3)
        t4 = t2 * t3
        return t4
# Inputs to the model
x1 = torch.randn(2, 4, requires_grad=True)
x2 = torch.randn(2, 4)
x3 = torch.randn(2, 4, requires_grad=True)
