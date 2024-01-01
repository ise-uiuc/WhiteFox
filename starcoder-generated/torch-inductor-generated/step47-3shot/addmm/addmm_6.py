
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp, a=1, b=2, c=3):
        v1 = torch.mm(a * x2, c * inp)
        v2 = v1 + x1 * (a * b * c)
        return v2
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3, requires_grad=True)
