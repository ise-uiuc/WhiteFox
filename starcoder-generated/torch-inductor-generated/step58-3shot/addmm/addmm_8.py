
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, inp):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x2, x3)
        v3 = v2 + x1
        w1 = torch.mm(x1, x3)
        w2 = torch.mm(w1 + inp, x3)
        w3 = v1 + x2
        w4 = torch.mm(x3, x2)
        w5 = torch.mm(x3, x2)
        return [x1, w1, v1, w2, w3, v2, w4, w5]
# Inputs to the model
x1 = torch.randn(666, 666)
x2 = torch.randn(666, 666, requires_grad=True)
x3 = torch.randn(666, 666)
x4 = torch.randn(666, 666, requires_grad=True)
inp = torch.randn(666, 666, requires_grad=True)
