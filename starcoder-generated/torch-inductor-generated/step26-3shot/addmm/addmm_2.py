
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        t1 = torch.mm(x2, inp)
        t2 = t1.repeat(1,3)
        t3 = torch.mm(t2, t2)
        t4 = torch.mm(t3, x2)
        return x2 + t4
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
